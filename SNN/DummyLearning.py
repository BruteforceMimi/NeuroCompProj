import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import csv
import simplifiedSTDP as stdp

from itertools import combinations

my_seed = 112
np.random.seed(my_seed)

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


def error_func_freq(desired_L, desired_R, actual_L, actual_R, window_duration):
    L = np.array(actual_L)
    R = np.array(actual_R)
    L = np.reshape(L, (len(desired_L), -1))
    R = np.reshape(L, (len(desired_R), -1))

    # scaled the frew down with factor 0.1
    L_freq = (np.count_nonzero(L, axis=1) / window_duration) / 10
    R_freq = (np.count_nonzero(R, axis=1) / window_duration) / 10
    print(f"the min values are L: {min(L_freq)} and R: {min(R_freq)}")
    print(f"the max values are L: {max(L_freq)} and R: {max(R_freq)}")
    left = np.abs(desired_L - L_freq)
    right = np.abs(desired_R - R_freq)
    return np.mean((left + right) / 2)

def error_func(desired_L, desired_R, actual_L, actual_R, min, max):
    actual_L = denormalise(actual_L,min, max)
    actual_R = denormalise(actual_R, min, max)
    left = np.abs(desired_L - actual_L)
    right = np.abs(desired_R - actual_R)
    return np.mean((left + right) / 2)

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

def normalise(my_spikes_L, my_spikes_R):
    my_spikes_L = np.array(my_spikes_L)
    my_spikes_R = np.array(my_spikes_R)
    max_ter = max(max(my_spikes_L[:,0]), max(my_spikes_R[:,0]))
    max_dist = max(max(my_spikes_L[:,1]), max(my_spikes_R[:,1]))
    min_ter = min(min(my_spikes_L[:,0]), min(my_spikes_R[:,0]))
    min_dist = min(min(my_spikes_L[:,1]), min(my_spikes_R[:,1]))
    my_spikes_L[:,0] = 2 * (my_spikes_L[:,0] - min_ter)/(max_ter - min_ter) - 1
    my_spikes_L[:, 1] = 2 * (my_spikes_L[:, 1] - min_dist) / (max_dist - min_dist) - 1
    my_spikes_R[:, 0] = 2 * (my_spikes_R[:, 0] - min_ter) / (max_ter - min_ter) - 1
    my_spikes_R[:, 1] = 2 * (my_spikes_R[:, 1] - min_dist) / (max_dist - min_dist) - 1
    return my_spikes_L, my_spikes_R

def denormalise(output, min, max):
    new_output = min + ((output+1)*(max-min))/2
    return new_output

def read_data():
    left = []
    right = []

    target_L = []
    target_R = []

    with open('C:/Users/Zizi/Desktop/master/Neuromorphic computing/project/NeuroCompProj/SNN/data_36.csv',
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
        Dij = 0.002
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


with nengo.Network(label="STDP", seed= my_seed) as model:
    # train input, initially not connected
    nr_neurons = 10
    train_signal_generator = nengo.Node(nengo.processes.PresentInput([[1.], [0.], [0.], [0.], [0.]], 0.005))

    # sensory input
    my_spikes_L, my_spikes_R, target_freq_L, target_freq_R = read_data()
    my_spikes_L, my_spikes_R = normalise(my_spikes_L, my_spikes_R)
    # my_spikes_L = [[0, 0.5], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0., 0]]
    # my_spikes_R = [[0, 0.5], [1.0, 1.0], [0., 0], [1.0, 0.0], [0.5, 0.5]]
    # target_freq_L = [20, 1, 30, 13, 12]
    # target_freq_R = [34, 15, 10, 2, 30]
    process_L = nengo.processes.PresentInput(my_spikes_L, timing)
    input_node_L = nengo.Node(process_L)
    process_R = nengo.processes.PresentInput(my_spikes_R, timing)
    input_node_R = nengo.Node(process_R)
    # https://forum.nengo.ai/t/spike-train-input-to-a-snn-model/717/4

    inp_collector_lter = nengo.Ensemble(nr_neurons, dimensions=1)
    inp_collector_rter = nengo.Ensemble(nr_neurons, dimensions=1)
    inp_collector_lgoal = nengo.Ensemble(nr_neurons, dimensions=1)
    inp_collector_rgoal = nengo.Ensemble(nr_neurons, dimensions=1)

    # input
    input_a = nengo.Ensemble(nr_neurons, dimensions=1)
    input_b = nengo.Ensemble(nr_neurons, dimensions=1)
    input_c = nengo.Ensemble(nr_neurons, dimensions=1)
    input_d = nengo.Ensemble(nr_neurons, dimensions=1)

    # hidden
    hidden_a = nengo.Ensemble(nr_neurons, dimensions=1)
    hidden_b = nengo.Ensemble(nr_neurons, dimensions=1)
    hidden_c = nengo.Ensemble(nr_neurons, dimensions=1)

    # output
    output_a = nengo.Ensemble(nr_neurons, dimensions=1)
    output_b = nengo.Ensemble(nr_neurons, dimensions=1)

    outa_p = nengo.Probe(output_a)
    outb_p = nengo.Probe(output_b)
    ## connections  between the layers

    nengo.Connection(input_node_L[0], inp_collector_lter)
    nengo.Connection(input_node_L[1], inp_collector_lgoal)
    nengo.Connection(input_node_R[0], inp_collector_rter)
    nengo.Connection(input_node_R[1], inp_collector_rgoal)

    nengo.Connection(inp_collector_lter, input_a)
    nengo.Connection(inp_collector_rter, input_b)
    nengo.Connection(inp_collector_lgoal, input_c)
    nengo.Connection(inp_collector_rgoal, input_d)


    nengo.Connection(input_a, hidden_a,  solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_a, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_a, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, hidden_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, hidden_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, hidden_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))

    nengo.Connection(hidden_a, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(hidden_b, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(hidden_c, output_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(hidden_a, output_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(hidden_b, output_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(hidden_c, output_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))

    nengo.Connection(input_a, input_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_a, input_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_a, input_d, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, input_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, input_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_b, input_d, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, input_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, input_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_c, input_d, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, input_a, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, input_b, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))
    nengo.Connection(input_d, input_c, solver=nengo.solvers.LstsqL2(weights=True),
                     learning_rule_type=stdp.STDP(learning_rate=2e-6))

    nengo.Connection(hidden_a, hidden_b)
    nengo.Connection(hidden_b, hidden_a)
    nengo.Connection(hidden_b, hidden_c)
    nengo.Connection(hidden_c, hidden_b)
    nengo.Connection(hidden_c, hidden_a)
    nengo.Connection(hidden_a, hidden_c)
# parameters
nr_datapoints = len(target_freq_R)
duration = timing * nr_datapoints
error = 10
error_limit = 0.5
training_pairs = []

# pick the first pair
all_pairs = list(combinations([input_a, input_b, input_c, input_d], 2))
indx_pairs = [*range(len(all_pairs))]
pair = np.random.choice(indx_pairs, replace=False)
training_pairs.append(all_pairs[pair])
del all_pairs[pair]

min_N_pairs = 1
max_N_pairs = len(all_pairs)
N = 1

#range of the output
max = max(max(target_freq_L), max(target_freq_R))
min = min(min(target_freq_L), min(target_freq_R))

while error > error_limit:
    with nengo.Simulator(model, progress_bar=False, seed = my_seed) as sim:
        sim.clear_probes()
        sim.run(duration)

    # compute error by comparing the output to the target
    # note: we might want to think about what exactly the output represents
    # and how it relates to the target freqs
    # optionnaly, the output should be transformed somehow

    #new_error = error_func(target_freq_L, target_freq_R, sim.data[outa_p], sim.data[outb_p], timing)  #
    new_error = error_func(target_freq_L, target_freq_R, sim.data[outa_p], sim.data[outb_p], min, max)
    sim.clear_probes()
    current_N = N
    if new_error <= error:
        while N < current_N * 2 and N < max_N_pairs:
            all_pairs, training_pairs = add_pair(all_pairs, training_pairs)
            N = N + 1
        for pre_neuron, post_neuron in training_pairs:
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False, seed = my_seed) as sim:
                sim.run(0.030)
    else:
        while N > current_N / 2 and N > min_N_pairs:
            all_pairs, training_pairs = remove_pair(all_pairs, training_pairs)
            N = N - 1
        training_pairs = [p[::-1] for p in training_pairs]
        # remove one pair, unless at minimum
        for pre_neuron, post_neuron in training_pairs:
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False, seed = my_seed) as sim:
                sim.run(0.030)
    print(f"current N is {N} and current error is {new_error}")
    error = new_error
    model = transform_to_validate(model)

print("final error was", error)

t = sim.trange()
plot_decoded(t, sim.data)

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

'''
Dij = 0.001
current N is 1 and current error is 17.80948214209443
current N is 1 and current error is 18.839735857479127
current N is 2 and current error is 15.113035674813792
current N is 1 and current error is 15.622269791356675
current N is 2 and current error is 15.123116962661362
current N is 1 and current error is 15.721509706536313
current N is 1 and current error is 17.352813247871737
current N is 2 and current error is 15.8652835866782
current N is 4 and current error is 15.548200906657096
current N is 2 and current error is 16.275404121736234

Dij = 0.002
current N is 1 and current error is 17.80948214209443
current N is 1 and current error is 18.86600207067564
current N is 2 and current error is 15.131911747394872
current N is 1 and current error is 15.530480537971526
current N is 2 and current error is 15.227480073829973
current N is 1 and current error is 15.886080733673882
current N is 1 and current error is 17.389522473272667
current N is 2 and current error is 16.036254628483277
current N is 4 and current error is 15.515984029286036
current N is 2 and current error is 16.232966891178467

D= 0.002 again: stays exactly the same
current N is 1 and current error is 17.80948214209443
current N is 1 and current error is 18.86600207067564
current N is 2 and current error is 15.131911747394868
current N is 1 and current error is 15.530480537971526
current N is 2 and current error is 15.227480073829973
.....

D = 0.005, lr = 2e-6
current N is 1 and current error is 19.58910874211295
current N is 2 and current error is 18.229477969860092
current N is 4 and current error is 16.385131194162778
current N is 2 and current error is 16.83609661412086
current N is 4 and current error is 15.416535843087415
current N is 5 and current error is 14.617118216882277
current N is 2 and current error is 15.038392985068688
current N is 1 and current error is 15.921124346666337
current N is 1 and current error is 18.14110435951876
current N is 2 and current error is 14.76772871741147
current N is 1 and current error is 17.13062067324734
current N is 2 and current error is 15.067705341478208
current N is 1 and current error is 17.965793353664612
current N is 2 and current error is 14.876315775362276

Dij = 0.002 lr = 2e-6
current N is 1 and current error is 19.58910874211295
current N is 2 and current error is 18.18070351833143
current N is 4 and current error is 17.17857183903441
current N is 5 and current error is 15.206978788765827
current N is 2 and current error is 15.293037269730798
current N is 1 and current error is 15.885837797491503
current N is 1 and current error is 15.911608898340274
current N is 1 and current error is 16.652443257620295
current N is 1 and current error is 16.881572096739074
current N is 2 and current error is 15.397015341759852
current N is 4 and current error is 14.90125646757962
current N is 2 and current error is 17.878726450473494
current N is 4 and current error is 16.8130709814194
current N is 5 and current error is 15.565275129236314
current N is 2 and current error is 16.31565618875729
current N is 4 and current error is 15.172382719509061
current N is 2 and current error is 15.385548744959817
current N is 1 and current error is 15.99013386792874
current N is 2 and current error is 15.563394247573228
current N is 1 and current error is 21.947109457274742

learning phase switched off
current N is 1 and current error is 19.58910874211295
current N is 2 and current error is 18.18070351833143
current N is 4 and current error is 17.291965285971067
current N is 5 and current error is 15.19275006880757
current N is 2 and current error is 15.436337006874057
current N is 4 and current error is 14.914206647840539
current N is 2 and current error is 15.788606961332633
current N is 1 and current error is 16.444803085005827
current N is 1 and current error is 17.9512322286322
current N is 2 and current error is 14.788763892208705
current N is 1 and current error is 20.100665270838828
current N is 2 and current error is 15.086344519037437

'''