import itertools
from graph_nets import utils_np

def _graph_from_ts(ts):
    """ Construct a graph with nodes as ROIs
    Args: 
        ts: np.array((num_timepoints, num_rois)) timeseries matrix
    Return:
        data_dict: dictionary containing
            "globals",
            "nodes"
            "edges"
            "senders"
            "receivers"
    Notes:
    """
    num_nodes = ts.shape[1]
    edges = [e for e in itertools.combinations(range(num_nodes), 2)]
    num_edges = len(edges)
    data_dict = {}
    
    data_dict['nodes'] = ts.T
    data_dict['edges'] = [0] * num_edges
    data_dict['senders'] = [e[0] for e in edges]
    data_dict['receivers'] = [e[1] for e in edges]
    return data_dict
    
    
def _clip_graphs(X):
    """ Convert list of timeseries matrices to list of graphs
    Args:
        X: [num_subjs x num_clips] list of timeseries matrices
    Return:
        graphs_tuple: [num_subjs x num_clips] list of timeseries graphs
    Notes:
    TODO: describe graph
    """
    
    num_samples = len(X)
    data_dict_list = []
    
    for idx_sample in range(num_samples):
        data_dict_list.append(_graph_from_ts(X[idx_sample]))
    
    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)
    
    return graphs_tuple
    