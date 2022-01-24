import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import csv
import simplifiedSTDP as stdp

def neg_sum_func(x):
    return -(x[0]+x[1])

def sum_func(x):
    return (x[0]+x[1])

def error_func(x, y):
    return (x - y)

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

def error_func(desired_L, desired_R, actual_L, actual_R):
    left = np.abs(desired_L - actual_L)
    right = np.abs(desired_R - actual_R)
    return np.mean(left + right)

def read_data():

    left = []
    right = []

    target_L = []
    target_R = []

    with open('../data.csv', newline='') as csvfile:
        datafile = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datafile:
            sample_left = []
            sample_right = []
            sample_left.append(float(row[0])) #terrain1 left
            sample_left.append(float(row[2])) #distnace1 left
            sample_right.append(float(row[1])) #terrain1 right
            sample_right.append(float(row[3])) #distnace1 right

            left.append(sample_left)
            right.append(sample_right)
            target_L.append(float(row[4]))
            target_R.append(float(row[5]))


    return left, right, target_L, target_R

with nengo.Network(label="STDP") as model:
    timing = 0.1

    my_spikes_L, my_spikes_R, target_freq_L, target_freq_R = read_data()
    print(my_spikes_L)
    #my_spikes_L = [[0,0],[0,0]]
    process_L = nengo.processes.PresentInput(my_spikes_L, timing)
    input_node_L = nengo.Node(process_L)
    #my_spikes_R = [[0], [1]]
    process_R = nengo.processes.PresentInput(my_spikes_R, timing)
    input_node_R = nengo.Node(process_R)
    # https://forum.nengo.ai/t/spike-train-input-to-a-snn-model/717/4
    input_node_probe = nengo.Probe(input_node_L)

    inp_collector_l = nengo.Ensemble(2, dimensions=2)
    inp_collector_r = nengo.Ensemble(2, dimensions=2)

    # input
    input_a = nengo.Ensemble(1, dimensions=1)
    input_b = nengo.Ensemble(1, dimensions=1)
    input_c = nengo.Ensemble(1, dimensions=1)
    input_d = nengo.Ensemble(1, dimensions=1)

    # hidden
    hidden_a = nengo.Ensemble(1, dimensions=1)
    hidden_b = nengo.Ensemble(1, dimensions=1)
    hidden_c = nengo.Ensemble(1, dimensions=1)

    # output
    output_a = nengo.Ensemble(1, dimensions=1)
    output_b = nengo.Ensemble(1, dimensions=1)

    outa_p = nengo.Probe(output_a)
    outb_p = nengo.Probe(output_b)
    ## connections  between the layers

    nengo.Connection(input_node_L, inp_collector_l, synapse=0.05)

    nengo.Connection(input_node_R, inp_collector_r, synapse=0.05)

    nengo.Connection(inp_collector_l[0], input_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(inp_collector_r[0], input_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(inp_collector_l[1], input_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(inp_collector_r[1], input_d, solver=nengo.solvers.LstsqL2(weights=True),
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
    nengo.Connection(input_c, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_c, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_c, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_d, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_d, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(input_d, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))

    nengo.Connection(hidden_a, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(hidden_b, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(hidden_c, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(hidden_a, output_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(hidden_b, output_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-9))
    nengo.Connection(hidden_c, output_b, solver=nengo.solvers.LstsqL2(weights=True),
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
    sim.run(0.1)
    freq_a = np.sum(sim.data[outa_p] > 0, axis=0) / len(sim.data[outa_p])
    freq_b = np.sum(sim.data[outb_p] > 0, axis=0) / len(sim.data[outb_p])
t = sim.trange()
print("freq A", freq_a)
print("freq B", freq_b)

print(sim.data[input_node_probe])

plot_decoded(t, sim.data)