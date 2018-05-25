
import numpy as np
import itertools
import queue as Q

adj = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1],
])

adj_michigan = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
])

all_states = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
])

print "The adjacency matrix"
print adj

q = Q.PriorityQueue()
visited = {}

leafs = np.where(adj[0][1:] == 1)[0]

print "The leafs, or entry points, to the graph"
print leafs

""" Calculate the leafs and all its permutations """

to_ret = np.array([], dtype=int).reshape(0, adj.shape[0] - 1)

for l in range(0, len(leafs)+1):
    for subset in itertools.combinations(leafs, l):
        if len(subset) == 0:
            to_ret = np.vstack((to_ret, np.zeros(adj.shape[0] - 1, dtype=int)))
        else:
            tmp = np.zeros(adj.shape[0] - 1, dtype=int)
            for i in subset:
                tmp[i] = 1
            to_ret = np.vstack((to_ret, tmp))

""" Visit all the leafs and add to the priority queue """

for _, l in enumerate(leafs):

    # Set node as visited
    visited[l] = True

    # Get the index of all the following nodes from the current node.
    next_nodes = np.where(adj[l + 1][np.arange(len(adj[l + 1])) != l + 1]
                          == 1)[0]

    for _, n in enumerate(next_nodes):

        # Check if the node has been visited before, if not, add to que & visit
        if n not in visited:
            visited[n] = True
            q.put((np.sum(adj, axis=0)[n + 1], n))

""" Visit all the remaining nodes and add the remaining states """

while not q.empty():
    # Get next node to visit
    next_node = q.get()[1]

    # The nodes that needs to enabled to enable the current node
    nodes_enabled = np.where(np.delete(adj[:, next_node + 1],
                                       next_node + 1) == 1)[0] - 1

    # All states containing the nodes that needs to be enabled.
    states_enabled = to_ret[np.all(np.equal(to_ret[:, nodes_enabled],
                                            np.ones(len(nodes_enabled))),
                                   axis=1)]

    # Add current node to the states with the nodes that needs to be enabled.
    states_enabled[:, next_node] += 1

    # Concat the new states to the old ones.
    to_ret = np.vstack((to_ret, states_enabled))

    # Get the index of all the following nodes from the current node.
    tmp_new = np.where(adj[next_node + 1][np.arange(
        len(adj[next_node + 1])) != next_node + 1] == 1)[0]

    for _, n in enumerate(tmp_new):

        # Check if the node has been visited before, if not, add to que & visit
        if n not in visited:
            visited[n] = True
            q.put((np.sum(adj, axis=0)[n + 1],
                   n))

print "The state matrix"
print to_ret
print "The shape of the state matrix", to_ret.shape
