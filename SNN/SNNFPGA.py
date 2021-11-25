import nengo
import numpy as np
import simplifiedSTDP as stdp

from nengo_fpga.networks import FpgaPesEnsembleNetwork

with nengo.Network() as model:

    stdp_rule = stdp.STDP()

    # Input stimulus
    input_node_left = nengo.Node(1)
    input_node_right = nengo.Node(2)

    input_layer = nengo.Ensemble(n_neurons=100, dimensions=2)

    nengo.Connection(input_node_left, input_layer[0])
    nengo.Connection(input_node_right, input_layer[1])

    solv = nengo.solvers.LstsqL2(weights=True)

    hidden_layer = nengo.Ensemble(n_neurons=256, dimensions=2)
    nengo.Connection(input_layer, hidden_layer, solver = solv, learning_rule_type = stdp_rule)


    output_layer1 = nengo.Ensemble(n_neurons=25, dimensions=1)
    output_layer2 = nengo.Ensemble(n_neurons=25, dimensions=1)

    nengo.Connection(hidden_layer[0], output_layer1)
    nengo.Connection(hidden_layer[1], output_layer2)

with nengo.Simulator(model) as sim:
    sim.run(10)

print("Yey done")