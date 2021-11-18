import nengo
import numpy as np

from nengo_fpga.networks import FpgaPesEnsembleNetwork

with nengo.Network() as model:

    # Input stimulus
    input_node_left = nengo.Node(0)
    input_node_right = nengo.Node(0)

    input_layer = nengo.Ensemble(n_neurons=100, dimensions=2)

    nengo.Connection(input_node_left, input_layer[0])
    nengo.Connection(input_node_right, input_layer[1])


    hidden_layer = nengo.Ensemble(n_neurons=256, dimensions=2)
    nengo.Connection(input_layer, hidden_layer)


    output_layer1 = nengo.Ensemble(n_neurons=25, dimensions=1)
    output_layer2 = nengo.Ensemble(n_neurons=25, dimensions=1)

    nengo.Connection(hidden_layer[0], output_layer1)
    nengo.Connection(hidden_layer[1], output_layer2)

