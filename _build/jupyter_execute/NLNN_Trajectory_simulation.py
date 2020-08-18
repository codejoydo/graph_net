# Trajectory Simulation

from IPython.display import display, HTML

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules
from graph_nets.demos_tf2 import models 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sonnet as snt
import tensorflow as tf
import tensorflow_addons as tfa

from data_processing_utils import *
from cc_utils import _get_clip_labels
from NLNN import *
 
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()

# avoid tensorflow from pre-allocating entire memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

### Load data 

# Load dataset 
X, X_len, clip_y, num_subjs, num_clips = get_data()

# clip names
clip_name_to_idx = _get_clip_labels()
clip_idx_to_name = {v: k for k, v in clip_name_to_idx.items()}
clip_idx_to_name[0] = clip_idx_to_name[0][:-1] # get rid of run number in test-retest
clip_names = list(clip_idx_to_name.keys())

# # Pad each time-series with zeros to equalize lengths
# X = pad_data(X)

# Fix a clip for simulating its trajectory
clip_num = 4
X_clip = [X[i] for i in range(len(clip_y)) if clip_y[i] == clip_num]
y_clip = [clip_y[i] for i in range(len(clip_y)) if clip_y[i] == clip_num]

### Create graphs from data

# Create data 
X = []
y = []
# Length of memory/history of data-sequence 
k = 5
for idx_subj in range(num_subjs):
    x = X_clip[idx_subj]
    for idx_tp in range(k, x.shape[0]):
        x_tp = x[idx_tp - k : idx_tp, :]
        y_tp = x[idx_tp, :]
        X.append(x_tp)
        y.append(y_tp)

y = np.array(y)        
        
# Create graphs from data
prob_edge = 0.1
graphs_dict_list = clip_graphs(X, 
                               prob_edge=prob_edge)

# Performance evaluation

def model_train_test(batch_size, train_G, train_y, test_G, test_y):
    
    template = ("Train Accuracy: {:.3%}, " 
                "Test Accuracy: {:.3%}  ")
    
    # Create model
    num_nodes = train_G[0]["nodes"].shape[0] # 300
    k_layers = [16]
    model = NLNNProcessDecode(num_nodes, 
                              k_layers, 
                              num_processing_steps=0)

    # Define loss and gradient functions
    loss_object = tf.keras.losses.MeanSquaredError()

    def loss(model, x, y):
      y_ = model(x)
      return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # training
    num_train = train_y.shape[0]
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 100
    progbar = tf.keras.utils.Progbar(num_epochs)

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tfa.metrics.RSquare()

        # Training loop - using batches of <batch_size>
        for i in range(0, num_train, batch_size):

            # Optimize the model
            x = utils_tf.data_dicts_to_graphs_tuple(train_G[i : i + batch_size])
            y = train_y[i : i + batch_size, :]
            loss_value, grads = grad(model, x, y)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress: Add current batch loss
            train_loss_avg.update_state(loss_value)  
            # Compare predicted label to actual label
            train_accuracy.update_state(tf.keras.backend.flatten(y),
                                        tf.keras.backend.flatten(model(x)))
            #if int(i/batch_size)%10 == 0:
            print("Train_loss: {:.3f}, Train Accuracy: {:.3f}".format(loss_value,
                                                                     train_accuracy.result()))

        # Log training loss and accuracy
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_accuracy.result())
        
        # update progress bar
        progbar.update(epoch+1)
    
    # testing
    num_test = test_y.shape[0]
    test_accuracy = tfa.metrics.RSquare()

    for i in range(0, num_test, batch_size):
        x = utils_tf.data_dicts_to_graphs_tuple(test_G[i : i + batch_size])
        y = test_y[i : i + batch_size, :]

        test_accuracy.update_state(tf.keras.backend.flatten(y),
                                        tf.keras.backend.flatten(model(x)))

    
    print(template.format(train_accuracy.result(), test_accuracy.result()))

    return(train_accuracy_results[-1].numpy(), test_accuracy.result().numpy())

## Training

# Train, test split
num_X = len(X)
num_train = round(0.8 * num_X)
num_val = 0 * num_X
num_test = num_X - num_train - num_val

num_splits = 1
test_accuracy_results = []
for idx_split in range(num_splits):
    
    # Shuffle data for random train-test splits
    rand_idx = np.rand_perm = np.random.permutation(num_X)
    graphs_dict_list_perm = list(map(lambda i: graphs_dict_list[i], rand_idx))
    y_perm = y[rand_idx, :]
    
    # Create split
    train_G, train_y, val_G, val_y, test_G, test_y = train_val_test_split(graphs_dict_list_perm,
                                                                           y_perm,
                                                                           num_train,
                                                                           num_val,
                                                                           num_test)

    # evaluate model 
    _, test_acc = model_train_test(128, 
                                   train_G, 
                                   train_y, 
                                   test_G, 
                                   test_y)
    test_accuracy_results.append(test_acc)

print("Mean test set accuracy: {:.3%}".format(np.mean(test_accuracy_results)))

print(train_G[0].keys())
print(train_G[0]['nodes'].dtype)
print(len(train_G[0]['edges']))
print(len(train_G[0]['senders']))
print(len(train_G[0]['receivers']))



