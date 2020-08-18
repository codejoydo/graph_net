import numpy as np
import networkx as nx
from graph_nets import utils_np 

SEED = 1
num_nodes = 300
prob_edge = 0.0000
graph = nx.fast_gnp_random_graph(n=num_nodes,
                        p=prob_edge,
                        seed=SEED)


edges = [e for e in graph.edges]
edges

300 * 299 / 2

