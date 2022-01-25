import nengo
import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import csv
import simplifiedSTDP as stdp


def neg_sum_func(x):
    return -(x[0] + x[1])


def sum_func(x):
    return (x[0] + x[1])




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

    with open('C:/Users/Zizi/Desktop/master/Neuromorphic computing/project/NeuroCompProj/SNN/data.csv', newline='') as csvfile:
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

def transform_to_train(model, pre_neurons, post_neurons):
    with model:
        Dij = 0.001
        for conn in model.all_connections:
            # disconnect from sensory input
            if (conn.pre_obj is inp_collector_l or conn.pre_obj is inp_collector_r) and conn.transform != 0.0:
                #add connection to pre neurons
                if conn.post_obj in post_neurons:
                    nengo.Connection(train_signal_generator, conn.post_obj, synapse = 0.005+Dij)
                    model.connections.remove(conn)
                if conn.post_obj in pre_neurons:
                    nengo.Connection(train_signal_generator, conn.post_obj, synapse=0.005)
                    model.connections.remove(conn)
                else:
                    nengo.Connection(conn.pre_obj, conn.post_obj, transform = 0.0)
                    model.connections.remove(conn)

            # change connections to STDP
            elif isinstance(conn.pre_obj, nengo.Ensemble) and conn.learning_rule_type is None:
                nengo.Connection(conn.pre_obj, conn.post_obj, solver = nengo.solvers.LstsqL2(weights=True),learning_rule_type = stdp.STDP(learning_rate=2e-9))
                model.connections.remove(conn)


        # connect chosen training pair to training signal generating nodes
        # (make sure that one trained neuron gets the signal with a delay of Dij
        # I don't know which value that should be so for now I just guess something reasonable)


    return model

def transform_to_validate(model):
    with model:

        #change the connections to non-learning connections
        #reconnect with the sensory input
        #disconnect from the train signal generator
        pass

with nengo.Network(label="STDP") as model:
    timing = 0.1

    # train input, initially not connected
    train_signal_generator = nengo.Node(nengo.processes.PresentInput([[0.], [1.]], 0.01))


    #sensory input
    my_spikes_L, my_spikes_R, target_freq_L, target_freq_R = read_data()
    # my_spikes_L = [[0,0],[0,0]]
    process_L = nengo.processes.PresentInput(my_spikes_L, timing)
    input_node_L = nengo.Node(process_L)
    # my_spikes_R = [[0], [1]]
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

    nengo.Connection(input_node_L, inp_collector_l)

    nengo.Connection(input_node_R, inp_collector_r)

    nengo.Connection(inp_collector_l[0], input_a,  solver=nengo.solvers.LstsqL2(weights=True),
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


    nengo.Connection(hidden_a, hidden_b)
    nengo.Connection(hidden_b, hidden_a)
    nengo.Connection(hidden_b, hidden_c)
    nengo.Connection(hidden_c, hidden_b)
    nengo.Connection(hidden_c, hidden_a)
    nengo.Connection(hidden_a, hidden_c)


    # ## WIP: calculate error through ensembles
    # target_L = nengo.Node(nengo.processes.PresentInput(target_freq_L, timing))
    # target_R = nengo.Node(nengo.processes.PresentInput(target_freq_R, timing))
    # error_L = nengo.Ensemble(10, dimensions = 1)
    # error_R = nengo.Ensemble(10, dimensions = 1)
    #
    # nengo.Connection(output_a, error_L, synapse=None, transform = -1)
    # nengo.Connection(target_L, error_L, synapse=None, transform = -1)
    #
    # nengo.Connection(output_b, error_R, synapse=None, transform = -1)
    # nengo.Connection(target_R, error_R, synapse=None, transform = -1)
    #
    # # -- inhibit errors after 40 seconds - gestolen van de tutorial
    # inhib = nengo.Node(lambda t: 2.0 if t > 40.0 else 0.0)
    # nengo.Connection(inhib, error_L.neurons, transform=[[-1]] * error_L.n_neurons)
    # nengo.Connection(inhib, error_R.neurons, transform=[[-1]] * error_R.n_neurons)
    #
    # # -- probes
    # target_p_L = nengo.Probe(target_L, synapse=0.01)
    # pre_p_L = nengo.Probe(input_node_L, synapse=0.01)
    # post_L_p = nengo.Probe(output_a, synapse=0.01)
    # error_L_p = nengo.Probe(error_L, synapse=0.03)

duration = 0.1
error = 10
error_limit  = 1
pre_neurons = [input_a]
post_neurons = [input_c]
i = 0
while i< 1:
    with nengo.Simulator(model) as sim:
        sim.run(duration)
        freq_a = np.sum(sim.data[outa_p] > 0, axis=0) / len(sim.data[outa_p])
        freq_b = np.sum(sim.data[outb_p] > 0, axis=0) / len(sim.data[outb_p])
    #compute error by comparing the output to the target
    # note: we might want to think about what exactly the output represents
    # and how it relates to the target freqs
    # optionnaly, the output should be transformed somehow
    new_error = error_func(target_freq_L, target_freq_R, sim.data[outa_p], sim.data[outb_p])#

    sim.data.reset()

    if new_error <= error:
        model = transform_to_train(model, pre_neurons, post_neurons)
    else:
        #reverse the training signal.
        #TODO: make the pairs separately instead of grouping them like that
        i = pre_neurons.copy()
        pre_neurons = post_neurons.copy()
        post_neurons = i
        model = transform_to_train(model, pre_neurons, post_neurons)


    with nengo.Simulator(model) as sim:
        sim.run(duration)
        freq_a = np.sum(sim.data[outa_p] > 0, axis=0) / len(sim.data[outa_p])
        freq_b = np.sum(sim.data[outb_p] > 0, axis=0) / len(sim.data[outb_p])
        i+=1
t = sim.trange()
print("freq A", freq_a)
print("freq B", freq_b)
print(sim.data[outa_p])
print(sim.data[outb_p])
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