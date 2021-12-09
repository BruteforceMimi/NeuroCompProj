import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt

import simplifiedSTDP as stdp

def neg_sum_func(x):
    return -(x[0]+x[1])

def sum_func(x):
    return (x[0]+x[1])

def plot_decoded(t, data, xlim_tuple=None):
    if xlim_tuple:
        x_start, x_end = xlim_tuple
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    if xlim_tuple:
        plt.xlim(x_start, x_end)
    plt.plot(t, data[outa_p].T[0], label="Output Left")
    plt.plot(t, data[outb_p].T[0], label="Output  Right")
    plt.ylabel("Decoded output")
    plt.legend(loc="best")

    plt.show()


with nengo.Network(label="STDP") as model:
    noise_period = 60
    cutoff_freq = 5
    stim_left_t = nengo.Node(WhiteSignal(noise_period, high=cutoff_freq), size_out=1)
    stim_left_o = nengo.Node(WhiteSignal(noise_period, high=cutoff_freq), size_out=1)
    stim_right_t = nengo.Node(WhiteSignal(noise_period, high=cutoff_freq), size_out=1)
    stim_right_o = nengo.Node(WhiteSignal(noise_period, high=cutoff_freq), size_out=1)

    inp_collector_l = nengo.Ensemble(1,dimensions = 1)
    inp_collector_r = nengo.Ensemble(1,dimensions = 1)


    #input
    input_a = nengo.Ensemble(1,dimensions = 1)
    input_b = nengo.Ensemble(1,dimensions = 1)

    #hidden
    hidden_a = nengo.Ensemble(1, dimensions = 1)
    hidden_b = nengo.Ensemble(1, dimensions = 1)
    hidden_c = nengo.Ensemble(1, dimensions = 1)

    #output
    output_a = nengo.Ensemble(1, dimensions = 1)
    output_b = nengo.Ensemble(1, dimensions = 1)

    outa_p = nengo.Probe(output_a)
    outb_p = nengo.Probe(output_b)

    ## connections  between the layers

    nengo.Connection(stim_left_t, inp_collector_l)
    nengo.Connection(stim_left_o, inp_collector_l)

    nengo.Connection(stim_right_t, inp_collector_r)
    nengo.Connection(stim_right_o, inp_collector_r)

    nengo.Connection(inp_collector_l, input_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(inp_collector_r, input_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))

    nengo.Connection(input_a, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_a, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_a, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_b, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_b, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_b, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))

    nengo.Connection( hidden_a, output_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection( hidden_b, output_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection( hidden_c, output_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection( hidden_a, output_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection( hidden_b, output_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection( hidden_c, output_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))

    ## connections within the layers
    nengo.Connection(input_a, input_b, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_b, input_a, solver=nengo.solvers.LstsqL2(weights=True),
        learning_rule_type=stdp.STDP(learning_rate=2e-9))


    nengo.Connection(hidden_a, hidden_b)
    nengo.Connection(hidden_b, hidden_a)
    nengo.Connection(hidden_b, hidden_c)
    nengo.Connection(hidden_c, hidden_b)
    nengo.Connection(hidden_c, hidden_a)
    nengo.Connection(hidden_a, hidden_c)

    nengo.Connection(output_a, output_b)
    nengo.Connection(output_b, output_a)



with nengo.Simulator(model) as sim:
    sim.run(3.0)
t = sim.trange()


plot_decoded(t, sim.data)
