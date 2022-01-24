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

    left_terrain = []
    right_terrain = []
    left_distance = []
    right_distance = []
    target_L = []
    target_R = []

    with open('../data.csv', newline='') as csvfile:
        datafile = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datafile:
            left_terrain = row[0]
            right_terrain = row[1]
            left_distance = row[2]
            right_distance = row[3]
            target_L = row[4]
            target_R = row[5]


    return [left_terrain, left_distance], [right_terrain, right_distance], target_L, target_R

with nengo.Network(label="STDP") as model:

    my_spikes_L, my_spikes_R, target_freq_L, target_freq_R = read_data()
    #my_spikes_L = [[0],[0]]
    input_node_L = nengo.Node(nengo.processes.PresentInput(my_spikes_L, 0.001))
    #my_spikes_R = [[0], [1]]
    input_node_R = nengo.Node(nengo.processes.PresentInput(my_spikes_R, 0.001))
    # https://forum.nengo.ai/t/spike-train-input-to-a-snn-model/717/4

    # stim_left_t = nengo.Node(output=1 , size_out=1)
    # stim_left_o = nengo.Node(output =1, size_out=1)
    # stim_right_t = nengo.Node(output = 1, size_out=1)
    # stim_right_o = nengo.Node(output = 1, size_out=1)

    inp_collector_l = nengo.Ensemble(8, dimensions = 1)
    inp_collector_r = nengo.Ensemble(8, dimensions = 1)

    dij_node = nengo.Node([1])

    Dij = dij_node.output[0]
    print(Dij)

    #input
    input_a = nengo.Ensemble(1, dimensions = 1)
    input_b = nengo.Ensemble(1, dimensions = 1)

    #hidden
    hidden_a = nengo.Ensemble(1, dimensions = 1)
    hidden_b = nengo.Ensemble(1, dimensions = 1)
    hidden_c = nengo.Ensemble(1, dimensions = 1)

    #output
    output_a = nengo.Ensemble(1, dimensions = 1)
    output_b = nengo.Ensemble(1, dimensions = 1)

    outa_p = nengo.Probe(output_a)
    outb_p = nengo.Probe(output_b)
    output_dij = nengo.Probe(dij_node)
    ## connections  between the layers



    nengo.Connection(input_node_L, inp_collector_l, synapse =0.05 )

    nengo.Connection(input_node_R, inp_collector_r, synapse = 0.05 )


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


sim = nengo.Simulator(model)
sim.run_steps(500)
freq_a = np.sum(sim.data[outa_p]>0, axis =0)/len(sim.data[outa_p])
freq_b = np.sum(sim.data[outb_p] > 0, axis=0)/len(sim.data[outb_p])
print("freq A",freq_a)
print("freq B",freq_b)

print("new sim")
sim.run(1.0)
freq_a = np.sum(sim.data[outa_p]>0, axis =0)/len(sim.data[outa_p])
freq_b = np.sum(sim.data[outb_p] > 0, axis=0)/len(sim.data[outb_p])
print("freq A",freq_a)
print("freq B",freq_b)

with nengo.Simulator(model) as sim:
    sim.run(1.0)
    freq_a = np.sum(sim.data[outa_p]>0, axis =0)
    freq_b = np.sum(sim.data[outb_p] > 0, axis=0)
t = sim.trange()
print("freq A",freq_a)
print("freq B",freq_b)




plot_decoded(t, sim.data)
