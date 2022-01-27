import nengo
import numpy as np
import simplifiedSTDP as stdp
import matplotlib.pyplot as plt


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
    #Check if shape matches, for now it only works with squares
    n1 = len(layer_plane1)
    print(n1)
    m1 = len(layer_plane1)

    n2 = len(layer_plane2)
    m2 = len(layer_plane2)

    for i in range(n1):
        nengo.Connection(layer_plane1[i][0], layer_plane2[i][-1], solver=solver, learning_rule_type=learning_rule)




with nengo.Network() as model:

    stdp_rule = stdp.STDP()
    solv = nengo.solvers.LstsqL2(weights=True)

    test_input_layer1 = create_network_layer(5, 5, stdp_rule, solv)
    test_input_layer2 = create_network_layer(5, 5, stdp_rule, solv)

    connect_layers(test_input_layer1, test_input_layer2, stdp_rule, solv)

    test_hidden_layer = create_network_layer(16, 16, stdp_rule, solv)

    connect_layers(test_input_layer1, test_hidden_layer, stdp_rule, solv)
    connect_layers(test_input_layer2, test_hidden_layer, stdp_rule, solv)

    test_output_layer1 = create_network_layer(5, 5, stdp_rule, solv)
    test_output_layer2 = create_network_layer(5, 5, stdp_rule, solv)

    # Input stimulus
    input_node_left = nengo.Node(1)
    input_node_right = nengo.Node(0)

    input_layer = nengo.Ensemble(n_neurons=100, dimensions=2)

    nengo.Connection(input_node_left, input_layer[0])
    nengo.Connection(input_node_right, input_layer[1])

    hidden_layer = nengo.Ensemble(n_neurons=256, dimensions=2)
    conn = nengo.Connection(input_layer, hidden_layer, solver = solv, learning_rule_type = stdp_rule)

    output_layer1 = nengo.Ensemble(n_neurons=25, dimensions=1)
    output_layer2 = nengo.Ensemble(n_neurons=25, dimensions=1)

    nengo.Connection(hidden_layer[0], output_layer1)
    nengo.Connection(hidden_layer[1], output_layer2)

    hidden_probe = nengo.Probe(conn.learning_rule, synapse=0.01)
    output1_probe = nengo.Probe(output_layer1)
    output2_probe = nengo.Probe(output_layer2)

with nengo.Simulator(model) as sim:
    sim.run(5)

print(sim.data[hidden_probe][2])
plt.plot(sim.data[hidden_probe][2], color="g", label="delta")
plt.show()
plt.plot(sim.data[output1_probe], color="r", label="output1")
plt.show()
plt.plot(sim.data[output2_probe], color="b", label="output2")
plt.show()
print("Yey done")