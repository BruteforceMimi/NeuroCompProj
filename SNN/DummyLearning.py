import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import csv
import simplifiedSTDP as stdp


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


with nengo.Network(label="STDP") as model:
    timing = 0.060

    # train input, initially not connected
    train_signal_generator = nengo.Node(nengo.processes.PresentInput([[0.], [1.]], 0.01))

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

    nengo.Connection(input_node_L[0], inp_collector_lter)
    nengo.Connection(input_node_L[1], inp_collector_lgoal)
    nengo.Connection(input_node_R[0], inp_collector_rter)
    nengo.Connection(input_node_R[1], inp_collector_rgoal)

    nengo.Connection(inp_collector_lter, input_a)
    nengo.Connection(inp_collector_rter, input_b)
    nengo.Connection(inp_collector_lgoal, input_c)
    nengo.Connection(inp_collector_rgoal, input_d)

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

    nengo.Connection(hidden_a, hidden_b)
    nengo.Connection(hidden_b, hidden_a)
    nengo.Connection(hidden_b, hidden_c)
    nengo.Connection(hidden_c, hidden_b)
    nengo.Connection(hidden_c, hidden_a)
    nengo.Connection(hidden_a, hidden_c)

duration = 1.3
error = 10
error_limit = 0.5
pre_neurons = [input_a]
post_neurons = [input_c]
index = 0

while error > error_limit:
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.data.reset()
        sim.run(0.3)

    # compute error by comparing the output to the target
    # note: we might want to think about what exactly the output represents
    # and how it relates to the target freqs
    # optionnaly, the output should be transformed somehow
    new_error = error_func(target_freq_L, target_freq_R, sim.data[outa_p], sim.data[outb_p],  0.060)  #
    print(new_error)

    sim.data.reset()

    if new_error <= error:
        for pre_neuron, post_neuron in zip(pre_neurons, post_neurons):
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False) as sim:
                sim.run(0.01)
    else:
        i = pre_neurons.copy()
        pre_neurons = post_neurons.copy()
        post_neurons = i
        for pre_neuron, post_neuron in zip(pre_neurons, post_neurons):
            model = transform_to_train(model, pre_neuron, post_neuron)
            with nengo.Simulator(model, progress_bar=False) as sim:
                sim.run(0.01)

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
