import nengo
import numpy as np
import warnings

from nengo import Ensemble
from nengo.builder.connection import slice_signal
from nengo.builder.learning_rules import get_post_ens, build_or_passthrough
from nengo.builder.operator import Reset, DotInc
from nengo.ensemble import Neurons
from nengo.processes import WhiteSignal

from nengo.config import SupportDefaultsMixin
from nengo.exceptions import ValidationError, BuildError
from nengo.params import Default, FrozenObject, IntParam, NumberParam, Parameter
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.numpy import is_iterable
from nengo.learning_rules import LearningRuleType
from nengo.builder import Builder, Signal
from nengo.builder.operator import Operator


class PES_2(LearningRuleType):
    """Prescribed Error Sensitivity learning rule.
    Modifies a connection's decoders to minimize an error signal provided
    through a connection to the connection's learning rule.
    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    """

    modifies = "decoders"
    probeable = ("error", "activities", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-4)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default):
        super().__init__(learning_rate, size_in="post_state")
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                "This learning rate is very high, and can result "
                "in floating point errors from too much current."
            )

        self.pre_synapse = pre_synapse

class SimPES_2(Operator):
    r"""Calculate connection weight change according to the PES rule.

    Implements the PES learning rule of the form

    .. math:: \Delta \omega_{ij} = \frac{\kappa}{n} e_j a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`n` is the number of presynaptic neurons
    * :math:`e_j` is the error for the jth output dimension, and
    * :math:`a_i` is the activity of a presynaptic neuron.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, error]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, error, delta, learning_rate, tag=None):
        super().__init__(tag=tag)

        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def error(self):
        return self.reads[1]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.delta)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        error = signals[self.error]
        delta = signals[self.delta]
        n_neurons = pre_filtered.shape[0]
        alpha = -self.learning_rate * dt / n_neurons

        def step_simpes():
            np.outer(alpha * error, pre_filtered, out=delta)

        return step_simpes

@Builder.register(PES_2)
def build_pes(model, pes, rule):
    """Builds a `.PES` object into a model.

    Calls synapse build functions to filter the pre activities,
    and adds a `.SimPES` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    pes : PES
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.PES` instance.
    """

    conn = rule.connection

    # Create input error signal
    error = Signal(shape=rule.size_in, name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough(
        model,
        pes.pre_synapse,
        slice_signal(
            model,
            model.sig[conn.pre_obj]["out"],
            conn.pre_slice,
        )
        if isinstance(conn.pre_obj, Neurons)
        else model.sig[conn.pre_obj]["out"],
    )

    if conn._to_neurons:
        # multiply error by post encoders to get a per-neuron error
        #   i.e. local_error = dot(encoders, error)
        post = get_post_ens(conn)
        if not isinstance(conn.post_slice, slice):
            raise BuildError(
                "PES learning rule does not support advanced indexing on non-decoded "
                "connections"
            )

        encoders = model.sig[post]["encoders"]
        # slice along neuron dimension if connecting to a neuron object, otherwise
        # slice along state dimension
        encoders = (
            encoders[:, conn.post_slice]
            if isinstance(conn.post_obj, Ensemble)
            else encoders[conn.post_slice, :]
        )

        local_error = Signal(shape=(encoders.shape[0],))
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, error, local_error, tag="PES:encode"))
    else:
        local_error = error

    model.add_op(SimPES_2(acts, local_error, model.sig[rule]["delta"], pes.learning_rate))

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts

#define random network for testing
def input_func(t):
    return [np.sin(t * 2*np.pi), np.cos(t * 2*np.pi)]

with nengo.Network(label="PES learning") as model:
    # Randomly varying input signal
    stim = nengo.Node(WhiteSignal(60, high=5), size_out=1)

    # Connect pre to the input signal
    pre = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(stim, pre)
    post = nengo.Ensemble(100, dimensions=1)

    # When connecting pre to post,
    # create the connection such that initially it will
    # always output 0. Usually this results in connection
    # weights that are also all 0.
    conn = nengo.Connection(
        pre,
        post,
        function=lambda x: [0],
        learning_rule_type=PES_2(learning_rate=2e-4),
    )

    # Calculate the error signal with another ensemble
    error = nengo.Ensemble(100, dimensions=1)

    # Error = actual - target = post - pre
    nengo.Connection(post, error)
    nengo.Connection(pre, error, transform=-1)

    # Connect the error into the learning rule
    nengo.Connection(error, conn.learning_rule)

    stim_p = nengo.Probe(stim)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    error_p = nengo.Probe(error, synapse=0.01)

with nengo.Simulator(model) as sim:
   sim.run(10)

print("help")