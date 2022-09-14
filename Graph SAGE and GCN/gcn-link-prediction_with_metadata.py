
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.0.0")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.0.0, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None


# In[2]:


import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding


from tensorflow import keras
import networkx as nx
import pandas as pd
import os
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
import pickle
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the CORA network data

# (See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)

# In[4]:


data_dir = "DBLP"
edgelist = pd.read_csv(
    os.path.join(data_dir, "edgeList.txt"),
    sep="\t",
    header=None,
    names=["source", "target"],
)
edgelist["label"] = "cites"  # set the edge type


feature_names = ["w_{}".format(ii) for ii in range(2476)]
node_column_names = feature_names + ["subject", "year"]
node_data = pd.read_csv(
    os.path.join(data_dir, "content.txt"), sep="\t", header=None, names=node_column_names
)
'''
print(node_data.columns)
node_data.drop(['subject', 'year'], axis=1)
'''

G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(G_all_nx,"paper","label")
all_node_features = node_data[feature_names]
G = sg.StellarGraph.from_networkx(G_all_nx, node_features=all_node_features)


# In[5]:

print(G.info())


# We aim to train a link prediction model, hence we need to prepare the train and test sets of links and the corresponding graphs with those links removed.
# 
# We are going to split our input graph into a train and test graphs using the EdgeSplitter class in `stellargraph.data`. We will use the train graph for training the model (a binary classifier that, given two nodes, predicts whether a link between these two nodes should exist or not) and the test graph for evaluating the model's performance on hold out data.
# Each of these graphs will have the same number of nodes as the input graph, but the number of links will differ (be reduced) as some of the links will be removed during each split and used as the positive samples for training/testing the link prediction classifier.

# From the original graph G, extract a randomly sampled subset of test edges (true and false citation links) and the reduced graph G_test with the positive test edges removed:

# In[6]:


# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(edge_labels_test)
print(edge_ids_test)

# The reduced graph G_test, together with the test ground truth set of links (edge_ids_test, edge_labels_test), will be used for testing the model.
# 
# Now repeat this procedure to obtain the training data for the model. From the reduced graph G_test, extract a randomly sampled subset of train edges (true and false citation links) and the reduced graph G_train with the positive train edges removed:

# In[7]:


# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)


# G_train, together with the train ground truth set of links (edge_ids_train, edge_labels_train), will be used for training the model.

# ## Creating the GCN link model

# Next, we create the link generators for the train and test link examples to the model. The link generators take the pairs of nodes (`citing-paper`, `cited-paper`) that are given in the `.flow` method to the Keras model, together with the corresponding binary labels indicating whether those pairs represent true or false links.
# 
# The number of epochs for training the model:

# In[8]:


epochs = 50


# For training we create a generator on the `G_train` graph, and make an iterator over the training links using the generator's `flow()` method:

# In[9]:


train_gen = FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)


# In[10]:


test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)


# Now we can specify our machine learning model, we need a few more parameters for this:
# 
#  * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GCN layers with 16-dimensional hidden node features at each layer.
#  * `activations` is a list of activations applied to each layer's output
#  * `dropout=0.3` specifies a 30% dropout at each layer. 

# We create a GCN model as follows:

# In[11]:


gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)


# To create a Keras model we now expose the input and output tensors of the GCN model for link prediction, via the `GCN.in_out_tensors` method:

# In[12]:


x_inp, x_out = gcn.in_out_tensors()


# Final link classification layer that takes a pair of node embeddings produced by the GCN model, applies a binary operator to them to produce the corresponding link embedding ('ip' for inner product; other options for the binary operator can be seen by running a cell with `?LinkEmbedding` in it), and passes it through a dense layer:

# In[13]:


prediction = LinkEmbedding(activation="sigmoid", method="ip")(x_out)


# The predictions need to be reshaped from `(X, 1)` to `(X,)` to match the shape of the targets we have supplied above.

# In[14]:


prediction = keras.layers.Reshape((-1,))(prediction)


# Stack the GCN and prediction layers into a Keras model, and specify the loss

# In[15]:


model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)


# Evaluate the initial (untrained) model on the train and test set:

# In[16]:




init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


# Train the model:

# In[17]:


history = model.fit(
    train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
)


# Plot the training history:

# In[18]:


sg.utils.plot_history(history)


# Evaluate the trained model on test citation links:

# In[19]:


train_metrics = model.predict(train_flow)
test_metrics = model.predict(test_flow)
print(test_metrics)

test_metrics_transformed=[]

for i in test_metrics:
    for j in i:
        if j>0.5:
            test_metrics_transformed.append(1)
        else:
            test_metrics_transformed.append(0)
print(test_metrics_transformed)
from sklearn.metrics import f1_score,roc_curve, auc
f1=f1_score(edge_labels_test,test_metrics_transformed)
print("f1_score",f1)
fpr, tpr, thresholds = roc_curve(edge_labels_test, test_metrics_transformed)    
roc_auc = auc(fpr, tpr)
print("roc_auc",roc_auc)

'''
pos_edges=[]
neg_edges=[]
for pos in range(0,len(test_metrics_transformed)):
    if test_metrics_transformed[pos] == 1:
        pos_edges.append(G_test.edges()[pos])
    else:
        neg_edges.append(G_test.edges()[pos])
'''

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)
print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

'''
pickle.dump(pos_edges,open("pos_edges.p","wb"))
pickle.dump(neg_edges,open("neg_edges.p","wb"))
pickle.dump(test_metrics_transformed,open("test_metrics_transformed.p","wb"))
pickle.dump(G_test.edges(),open("G_test.p","wb"))
print(len(neg_edges))
'''