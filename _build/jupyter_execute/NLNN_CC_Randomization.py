# Breaking Down Classifier

from IPython.display import display, HTML

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sonnet as snt
import tensorflow as tf

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

# Pad each time-series with zeros to equalize lengths
X = pad_data(X)



### Create graphs from data

# Create graphs from data
prob_edge = 0.0
graphs_dict_list = clip_graphs(X, 
                               prob_edge=prob_edge)

# Convert clip labels to one-hot vectors
clip_y_oh = tf.one_hot(clip_y, num_clips).numpy()


### Performance evaluation
### Cross-validation

def model_train_test(batch_size, train_G, train_y, test_G, test_y):
    
    template = ("Train Accuracy: {:.3%}, " 
                "Test Accuracy: {:.3%}  ")
    
    # Create model
    model = NLNNClassifer(num_nodes=300, 
                          k_linear=16, 
                          num_classes=num_clips)

    # Define loss and gradient functions
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def loss(model, x, y):
      # training=training is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      y_ = model(x)
      return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # training
    num_train = train_y.shape[0]
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 10
    progbar = tf.keras.utils.Progbar(num_epochs)

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_loss_avg = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of <batch_size>
        for i in range(0, num_train, batch_size):

            # Optimize the model
            x = utils_tf.data_dicts_to_graphs_tuple(train_G[i : i + batch_size])
            yt = model(x)
            y = train_y[i : i + batch_size, :]
            loss_value, grads = grad(model, x, y)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress: Add current batch loss
            train_loss_avg.update_state(loss_value)  
            # Compare predicted label to actual label
            train_accuracy.update_state(y, model(x))

        # Log training loss and accuracy
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_accuracy.result())
        
        # update progress bar
        progbar.update(epoch+1)
    
    # testing
    num_test = test_y.shape[0]
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for i in range(0, num_test, batch_size):
        x = utils_tf.data_dicts_to_graphs_tuple(test_G[i : i + batch_size])
        y = test_y[i : i + batch_size, :]
        logits = model(x)

        test_accuracy.update_state(logits, y)

    
    print(template.format(train_accuracy.result(), test_accuracy.result()))

    return(train_accuracy_results[-1].numpy(), test_accuracy.result().numpy())

### With shuffled labels

# Total number of movies watched by each participant
# Because there are four runs of test-retest (clip #0)
num_movies = 18

# Train, test split
# Split participants
num_train = 100 * num_movies
num_val = 0 * num_movies
num_test = 76 * num_movies

num_splits = 10
test_accuracy_results_random_labels = []
for idx_split in range(num_splits):
    
    # Shuffle data for random train-test splits
    subj_perm_idx = np.rand_perm = np.random.permutation(num_subjs)
    rand_idx = [list(range(i * num_movies, (i + 1) * num_movies)) for i in subj_perm_idx]
    rand_idx = sum(rand_idx, [])
    graphs_dict_list_perm = list(map(lambda i: graphs_dict_list[i], rand_idx))
    clip_y_oh_perm = clip_y_oh[rand_idx, :]
    
    # Create split
    train_G, train_y, val_G, val_y, test_G, test_y = train_val_test_split(graphs_dict_list_perm,
                                                                           clip_y_oh_perm,
                                                                           num_train,
                                                                           num_val,
                                                                           num_test)

    # Shuffle training labels only
    rand_perm = np.random.permutation(train_y.shape[0])
    train_y = train_y[rand_perm, :]
    
    # evaluate model 
    _, test_acc = model_train_test(8, 
                                   train_G, 
                                   train_y, 
                                   test_G, 
                                   test_y)
    test_accuracy_results_random_labels.append(test_acc)

print("Mean test set accuracy: {:.3%}".format(np.mean(test_accuracy_results_random_labels)))

### With true labels

# Total number of movies watched by each participant
# Because there are four runs of test-retest (clip #0)
num_movies = 18

# Train, test split
# Split participants
num_train = 100 * num_movies
num_val = 0 * num_movies
num_test = 76 * num_movies

num_splits = 10
test_accuracy_results = []
for idx_split in range(num_splits):
    
    # Shuffle data for random train-test splits
    subj_perm_idx = np.rand_perm = np.random.permutation(num_subjs)
    rand_idx = [list(range(i * num_movies, (i + 1) * num_movies)) for i in subj_perm_idx]
    rand_idx = sum(rand_idx, [])
    graphs_dict_list_perm = list(map(lambda i: graphs_dict_list[i], rand_idx))
    clip_y_oh_perm = clip_y_oh[rand_idx, :]
    
    # Create split
    train_G, train_y, val_G, val_y, test_G, test_y = train_val_test_split(graphs_dict_list_perm,
                                                                           clip_y_oh_perm,
                                                                           num_train,
                                                                           num_val,
                                                                           num_test)
    
    # evaluate model 
    _, test_acc = model_train_test(8, 
                                   train_G, 
                                   train_y, 
                                   test_G, 
                                   test_y)
    test_accuracy_results.append(test_acc)

print("Mean test set accuracy: {:.3%}".format(np.mean(test_accuracy_results)))

### Cross-validation results

# Group data together
df = pd.DataFrame(columns = ["accuracy", "group"])
df["accuracy"] = test_accuracy_results + test_accuracy_results_random_labels
df["group"] = ["True"] * num_splits + ["Shuffled"] * num_splits

# Create distplot with custom bin_size
fig = px.violin(df, 
                y="accuracy", 
                x="group",
                box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
                color="group",
                hover_data=df.columns,
                title="Cross-validation accuracy<br>10 train-test splits per group<br>True labels vs Shuffled labels"
               )
fig.show()

