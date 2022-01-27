import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import csv
import simplifiedSTDP as stdp

from itertools import combinations

def findNeighbors(grid, x, y):
    if 0 < x < len(grid) - 1:
        xi = (0, -1, 1)   # this isn't first or last row, so we can look above and below
    elif x > 0:
        xi = (0, -1)      # this is the last row, so we can only look above
    else:
        xi = (0, 1)       # this is the first row, so we can only look below
    # the following line accomplishes the same thing as the above code but for columns
    yi = (0, -1, 1) if 0 < y < len(grid[0]) - 1 else ((0, -1) if y > 0 else (0, 1))
    for a in xi:
        for b in yi:
            if a == b == 0:  # this value is skipped using islice in the original code
                continue
            yield grid[x + a][y + b]


def create_network_layer(n, m, learning_rule, solver):
    grid = [[0]*n for i in range(m)]

    for i in range(n):
        for j in range(m):
            grid[i][j] = nengo.Ensemble(n_neurons=1, dimensions=1)

    for i in range(n):
        for j in range(m):
            neigh = list(findNeighbors(grid, i, j))
            for elem in neigh:
                nengo.Connection(grid[i][j], elem, solver=solver, learning_rule_type=learning_rule)

    return grid


def connect_layers(layer_plane1, layer_plane2, learning_rule, solver):
    n1 = len(layer_plane1)
    n2 = len(layer_plane2)


    for i in range(n1):
        for j in range(n2):
            nengo.Connection(layer_plane1[i][0], layer_plane2[j][-1], solver=solver, learning_rule_type=learning_rule)

def connect_layer_ensemble(ensemble, layer_plane, learning_rule, solver):
    n1 = len(layer_plane)

    for i in range(n1):
        nengo.Connection(layer_plane[i][0], ensemble, solver=solver, learning_rule_type=learning_rule)


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


def error_func(desired_L, desired_R, actual_L, actual_R, window_duration):
    L = np.array(actual_L)
    R = np.array(actual_R)
    L = np.reshape(L, (len(desired_L),-1))
    R = np.reshape(L, (len(desired_R), -1))

    #scaled the frew down with factor 0.1
    L_freq = (np.count_nonzero(L,axis = 1)/ window_duration)/10
    R_freq = (np.count_nonzero(R, axis=1) / window_duration)/10

    left = np.abs(desired_L - L_freq)
    right = np.abs(desired_R - R_freq)
    return np.mean((left + right)/2)

def add_pair(all_pairs, training_pairs):
    indx_pairs = [*range(len(all_pairs))]
    pair = np.random.choice(indx_pairs, replace=False)
    training_pairs.append(all_pairs[pair])
    del all_pairs[pair]
    return all_pairs, training_pairs

def remove_pair(all_pairs, training_pairs):
    indx_pairs = [*range(len(training_pairs))]
    pair = np.random.choice(indx_pairs, replace=False)
    all_pairs.append(training_pairs[pair])
    del training_pairs[pair]
    return all_pairs, training_pairs


def read_data():
    left = []
    right = []

    target_L = []
    target_R = []

    with open('C:/Users/Zizi/Desktop/master/Neuromorphic computing/project/NeuroCompProj/SNN/data.csv',
              newline='') as csvfile:
        datafile = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datafile:
            sample_left = []
            sample_right = []
            sample_left.append(float(row[0]))  # terrain1 left
            sample_left.append(float(row[2]))  # distnace1 left
            sample_right.append(float(row[1]))  # terrain1 right
            sample_right.append(float(row[3]))  # distnace1 right

            left.append(sample_left)
            right.append(sample_right)
            target_L.append(float(row[4]))
            target_R.append(float(row[5]))

    return left, right, target_L, target_R


def transform_to_train(model, pre_neuron, post_neuron):
    with model:
        Dij = 0.001
        to_remove = []
        for conn in model.all_connections:
            if conn.pre_obj is inp_collector_lgoal or conn.pre_obj is inp_collector_lter or conn.pre_obj is inp_collector_rgoal or conn.pre_obj is inp_collector_rter:
                to_remove.append(conn)
        for conn2 in to_remove:
            model.connections.remove(conn2)

        nengo.Connection(train_signal_generator, pre_neuron, synapse=0.005)
        nengo.Connection(train_signal_generator, post_neuron, synapse=0.005 + Dij)
    return model


def transform_to_validate(model):
    with model:
        # remove all trainging connections to input collectors
        to_remove = []
        for conn in model.all_connections:
            if conn.pre_obj is inp_collector_lgoal or conn.pre_obj is inp_collector_lter or conn.pre_obj is inp_collector_rgoal or conn.pre_obj is inp_collector_rter:
                to_remove.append(conn)
        for conn2 in to_remove:
            model.connections.remove(conn2)

        # add the sensor connections back to where they belong
        nengo.Connection(inp_collector_lter, input_a)
        nengo.Connection(inp_collector_rter, input_b)
        nengo.Connection(inp_collector_lgoal, input_c)
        nengo.Connection(inp_collector_rgoal, input_d)
    return model

timing = 0.060
with nengo.Network(label="STDP") as model:

    # train input, initially not connected
    train_signal_generator = nengo.Node(nengo.processes.PresentInput([[1.], [0.],[0.],[0.],[0.]], 0.005))

    # sensory input
    # my_spikes_L, my_spikes_R, target_freq_L, target_freq_R = read_data()
    my_spikes_L = [[0,0.5],[1.0,0.0],[0.5,0.5],[1.0,1.0],[0.,0]]
    my_spikes_R = [[0, 0.5], [1.0, 1.0], [0., 0], [1.0, 0.0], [0.5, 0.5]]
    target_freq_L = [20 , 1, 30, 13 , 12 ]
    target_freq_R = [ 34, 15, 10 ,2, 30]
    process_L = nengo.processes.PresentInput(my_spikes_L, timing)
    input_node_L = nengo.Node(process_L)
    process_R = nengo.processes.PresentInput(my_spikes_R, timing)
    input_node_R = nengo.Node(process_R)
    # https://forum.nengo.ai/t/spike-train-input-to-a-snn-model/717/4
    input_node_probe = nengo.Probe(input_node_L)

    inp_collector_lter = nengo.Ensemble(1, dimensions=1)
    inp_collector_rter = nengo.Ensemble(1, dimensions=1)
    inp_collector_lgoal = nengo.Ensemble(1, dimensions=1)
    inp_collector_rgoal = nengo.Ensemble(1, dimensions=1)


    stdp_rule = stdp.STDP()
    solv = nengo.solvers.LstsqL2(weights=True)

    input_layer1 = create_network_layer(5, 5, stdp_rule, solv)
    input_layer2 = create_network_layer(5, 5, stdp_rule, solv)

    connect_layer_ensemble(inp_collector_lgoal, input_layer1, stdp_rule, solv)
    connect_layer_ensemble(inp_collector_lter, input_layer1, stdp_rule, solv)

    connect_layer_ensemble(inp_collector_rgoal, input_layer2, stdp_rule, solv)
    connect_layer_ensemble(inp_collector_rter, input_layer2, stdp_rule, solv)

    hidden_layer = create_network_layer(16, 16, stdp_rule, solv)

    connect_layers(input_layer1, hidden_layer, stdp_rule, solv)
    connect_layers(input_layer2, hidden_layer, stdp_rule, solv)

    output_layer1 = create_network_layer(5, 5, stdp_rule, solv)
    output_layer2 = create_network_layer(5, 5, stdp_rule, solv)

    # input
    # input_a = nengo.Ensemble(1, dimensions=1)
    # input_b = nengo.Ensemble(1, dimensions=1)
    # input_c = nengo.Ensemble(1, dimensions=1)
    # input_d = nengo.Ensemble(1, dimensions=1)
    #
    # # hidden
    # hidden_a = nengo.Ensemble(1, dimensions=1)
    # hidden_b = nengo.Ensemble(1, dimensions=1)
    # hidden_c = nengo.Ensemble(1, dimensions=1)
    #
    # # output
    # output_a = nengo.Ensemble(1, dimensions=1)
    # output_b = nengo.Ensemble(1, dimensions=1)
    #
    # outa_p = nengo.Probe(output_a)
    # outb_p = nengo.Probe(output_b)
    # ## connections  between the layers
    #
    # nengo.Connection(input_node_L[0], inp_collector_lter)
    # nengo.Connection(input_node_L[1], inp_collector_lgoal)
    # nengo.Connection(input_node_R[0], inp_collector_rter)
    # nengo.Connection(input_node_R[1], inp_collector_rgoal)
    #
    # nengo.Connection(inp_collector_lter, input_a)
    # nengo.Connection(inp_collector_rter, input_b)
    # nengo.Connection(inp_collector_lgoal, input_c)
    # nengo.Connection(inp_collector_rgoal, input_d)
    #
    # nengo.Connection(input_a, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_a, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_a, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_b, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_b, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_b, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_c, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_c, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_c, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_d, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_d, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(input_d, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    #
    # nengo.Connection(hidden_a, output_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(hidden_b, output_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(hidden_c, output_a, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(hidden_a, output_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(hidden_b, output_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    # nengo.Connection(hidden_c, output_b, solver=nengo.solvers.LstsqL2(weights=True),
    #                  learning_rule_type=stdp.STDP(learning_rate=2e-9))
    #
    # nengo.Connection(hidden_a, hidden_b)
    # nengo.Connection(hidden_b, hidden_a)
    # nengo.Connection(hidden_b, hidden_c)
    # nengo.Connection(hidden_c, hidden_b)
    # nengo.Connection(hidden_c, hidden_a)
    # nengo.Connection(hidden_a, hidden_c)

#paramters
nr_datapoints = len(target_freq_R)
duration = timing * nr_datapoints
print("duration is ",duration)
error = 10
error_limit = 0.5
training_pairs = []

#pick the first pair
all_pairs = list(combinations([input_a,input_b, input_c, input_d],2))
indx_pairs = [*range(len(all_pairs))]
pair = np.random.choice(indx_pairs, replace = False)
training_pairs.append(all_pairs[pair])
del all_pairs[pair]

min_N_pairs = 1
max_N_pairs = len(all_pairs)
N = 1

while error > error_limit:
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.data.reset()
        sim.run(duration)

    # compute error by comparing the output to the target
    # note: we might want to think about what exactly the output represents
    # and how it relates to the target freqs
    # optionnaly, the output should be transformed somehow
    new_error = error_func(target_freq_L, target_freq_R, sim.data[outa_p], sim.data[outb_p],  timing)  #

    sim.data.reset()
    current_N = N
    if new_error <= error:
        while N < current_N*2 and N < max_N_pairs:
            all_pairs,training_pairs = add_pair(all_pairs,training_pairs)
            N = N+1
        for pre_neuron, post_neuron in training_pairs:
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False) as sim:
                sim.run(0.025)
    else:
        while N > current_N/2 and N > min_N_pairs:
            all_pairs,training_pairs = remove_pair(all_pairs,training_pairs)
            N = N-1
        training_pairs = [p[::-1] for p in training_pairs]
        #remove one pair, unless at minimum
        for pre_neuron, post_neuron in training_pairs:
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False) as sim:
                sim.run(0.025)
    print(f"current N is {N} and current error is {new_error}")
    error = new_error
    model = transform_to_validate(model)

print("final error was", error)

t = sim.trange()
#plot_decoded(t, sim.data)

# Gestolen van tutorial
# sl = slice(0, duration-1)
# t = sim.trange()[sl]
# plt.figure(figsize=(14, 12))
# plt.suptitle("")
# plt.subplot(4, 1, 1)
# plt.plot(t, sim.data[pre_p_L][sl], c="b")
# plt.legend(("Pre decoding",), loc="best")
# plt.subplot(4, 1, 2)
# plt.plot(t, sim.data[target_p_L][sl], c="k", label="Actual freq")
# plt.plot(t, sim.data[post_L_p][sl], c="r", label="Post decoding (Left)")
# plt.legend(loc="best")
# plt.subplot(4, 1, 3)
# plt.plot(t, sim.data[target_p_L][sl], c="k", label="Actual freq")
# plt.plot(t, sim.data[error_L_p][sl], c="r", label="Error")
# plt.legend(loc="best")
#
# plt.show()
