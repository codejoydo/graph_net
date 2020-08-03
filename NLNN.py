from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import _base
from graph_nets import blocks

import sonnet as snt
import tensorflow as tf

_NLNN_DEFAULT_EDGE_BLOCK_OPT = {
    "use_edges": False,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

_NLNN_DEFAULT_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

class NLNN(_base.AbstractModule):
  """Implementation of a Non Local Neural Network.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               edge_block_opt=_NLNN_DEFAULT_EDGE_BLOCK_OPT,
               node_block_opt=_NLNN_DEFAULT_NODE_BLOCK_OPT,
               name="nlnn"):
    """Initializes the GraphNetwork module.
    Args:
      edge_model_fn: A callable that will be passed to EdgeBlock to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see EdgeBlock for details).
      node_model_fn: A callable that will be passed to NodeBlock to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see NodeBlock for details).
      reducer: Reducer to be used by NodeBlock aggregate edges.
        Defaults to tf.math.unsorted_segment_sum. This will be overridden by
        the reducers specified in `node_block_opt`, if any.
      edge_block_opt: Additional options to be passed to the EdgeBlock. Can
        contain keys `use_edges`, `use_receiver_nodes`, `use_sender_nodes`. 
        By default, these are all True.
      node_block_opt: Additional options to be passed to the NodeBlock. Can
        contain the keys `use_received_edges`, `use_nodes`, `use_globals` (all
        set to True by default), `use_sent_edges` (defaults to False), and
        `received_edges_reducer`, `sent_edges_reducer` (default to `reducer`).
      name: The module name.
    """
    super(NLNN, self).__init__(name=name)
    edge_block_opt = modules._make_default_edge_block_opt(edge_block_opt)
    node_block_opt = modules._make_default_node_block_opt(node_block_opt, reducer)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn, **edge_block_opt)
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn, **node_block_opt)

  def _build(self, graph):
    """Connects the GraphNetwork.
    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. Depending on the block
        options, `graph` may contain `None` fields; but with the default
        configuration, no `None` field is allowed. Moreover, when using the
        default configuration, the features of each nodes, edges and globals of
        `graph` should be concatenable on the last dimension.
    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.
    """
    return self._node_block(self._edge_block(graph))


class NLNNClassifer(snt.Module):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, num_nodes, k_linear, num_classes=15):
    super(NLNNClassifer, self).__init__()
    self._num_nodes = num_nodes
    self._k_linear = k_linear
    self._network = NLNN(
        edge_model_fn=lambda: snt.nets.MLP([self._k_linear]),
        node_model_fn=lambda: snt.nets.MLP([self._k_linear]))
    self._fc = snt.Linear(num_classes)

  def __call__(self, inputs):
    outputs = self._network(inputs).nodes
    
    # outputs is [batch_size x num_nodes, num_feat] matrix 
    # reshape it into [batch_size, num_nodes, num_feat] ndarray
    outputs = tf.reshape(outputs, [-1, self._num_nodes, self._k_linear])
    
    # flatten [batch_size, num_nodes, num_feat] ndarray
    # into [batch_size, num_nodes x num_feat] matrix
    outputs = snt.flatten(outputs)
    
    outputs = self._fc(outputs)
    return outputs