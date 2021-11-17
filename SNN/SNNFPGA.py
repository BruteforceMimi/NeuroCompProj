import nengo
import numpy as np

from nengo_fpga.networks import FpgaPesEnsembleNetwork
## Testing if everything works

def input_func(t):
    return [np.sin(t * 2*np.pi), np.cos(t * 2*np.pi)]

with nengo.Network() as model:

    # Input stimulus
    input_node = nengo.Node(input_func)

    # "Pre" ensemble of neurons, and connection from the input
    ens_fpga = FpgaPesEnsembleNetwork('de1', n_neurons=50,
                                      dimensions=2,
                                      learning_rate=1e-4)
    nengo.Connection(input_node, ens_fpga.input)  # Note the added '.input'