import itertools
from graph_nets import utils_tf
from dataloader import _clip_class_df, _get_clip_seq 
import numpy as np
import networkx as nx

NX_SEED = 1

class ARGS():
    pass

def get_data():
    """ Wrapper for extracting
            - clip fMRI time series and 
            - clip indexes as labels
    """
    args = ARGS()
    args.roi = 300 # number of ROIs
    args.net = 7 # number of subnetworks (7 or 17)
    args.subnet = 'wb' #subnetwork; 'wb' if all subnetworks
    args.invert_flag = False # all-but-one subnetwork
    args.input_data = 'data/roi_ts'
    args.zscore = 1
    
    df = _clip_class_df(args)
    args.k_class = len(np.unique(df['y']))
    subject_list = df['Subject'].unique()
    X, X_len, clip_y = _get_clip_seq(df, subject_list, args)
    clip_y = np.array(clip_y)
    
    num_subjs = df.Subject.unique().shape[0]
    num_clips = np.unique(clip_y).shape[0]

    return (X, X_len, clip_y, num_subjs, num_clips)
    
def pad_data(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(r.shape[0] for r in M)

    Z = []
    for enu, row in enumerate(M):
        Z.append(np.zeros((maxlen, row.shape[1])))
        Z[enu][ : row.shape[0], ] += row 
    return Z

def train_val_test_split(G, y, num_train, num_val, num_test, num_clips, num_subjs):
    """ Split graphs_tuple and labels into train-val-test sets
    Args:
        G: graphs_tuple object
        y: labels
        num_train: number of training samples
        num_val: number of validation samples
        num_test: number of testing samples
        num_clips: total number of clips
        num_subjs: total number of subjects
    Return:
        train_G
        train_y
        val_G
        val_Y
        test_G
        test_Y
    """
    start_tr = 0
    end_tr = start_va = num_train
    end_va = start_te = num_train + num_val
    end_te = num_train + num_val + num_test
    
    train_G = utils_tf.get_graph(G, slice(start_tr, end_tr))
    train_y = y[start_tr : end_tr, :]
    
    val_G = utils_tf.get_graph(G, slice(start_va, end_va))
    val_y = y[start_va : end_va, :]
    
    test_G = utils_tf.get_graph(G, slice(start_te, end_te))
    test_y = y[start_te : end_te, :]
    
    return (train_G, train_y, val_G, val_y, test_G, test_y)

def _graph_from_ts(ts, edges):
    """ Construct a graph with nodes as ROIs
    Args: 
        ts: np.array((num_timepoints, num_rois)) timeseries matrix
        edges: [num_edges] list of edges
    Return:
        data_dict: dictionary containing
            "globals"
            "nodes"
            "edges"
            "senders"
            "receivers"
        See https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb
        for more details.
    Notes:
    The features of each node (ROI) is its entire activation time-series
    of dimension [max_length, ]. 
    Thus the nodes attributes of the graph is a matrix of size [num_rois, max_length].
    """
    num_nodes = ts.shape[1]
    # To reject hypothesis that model learns merely by clip length.
    ts = ts[0:84,:] 
    
    num_edges = len(edges)
    data_dict = {}
    
    data_dict['nodes'] = ts.T
    # Dummy edge attributes to facilitate graph construction. 
    data_dict['edges'] = [0] * num_edges 
    data_dict['senders'] = [e[0] for e in edges]
    data_dict['receivers'] = [e[1] for e in edges]
    return data_dict
    
    
def clip_graphs(X, prob_edge=0.1):
    """ Convert list of timeseries matrices to list of graphs
    Args:
        X: [num_subjs x num_clips] list of timeseries matrices
        prob_edge: probability threshold for generating a random graph
    Return:
        graphs_tuple: [num_subjs x num_clips] list of timeseries graphs
    """
    num_nodes = X[0].shape[1]
    graph = nx.fast_gnp_random_graph(n=num_nodes,
                        p=prob_edge,
                        seed=NX_SEED)
    edges = [e for e in graph.edges]
    
    num_samples = len(X)
    data_dict_list = []
    
    for idx_sample in range(num_samples):
        data_dict_list.append(_graph_from_ts(X[idx_sample], edges))
    
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(data_dict_list)
    
    return graphs_tuple




    
   