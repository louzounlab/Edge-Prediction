# Edge Prediction
Graph's edges classification by topological (and other) features with neural network.

## Installation
* Installation via pip: `pip install EdgeClassifier`.
* Installation with git: `git clone https://github.com/louzounlab/Edge-Prediction`.

## How to use?
```python
import networkx as nx
from EdgeClassifier.edge_classifier import EdgeClassifier

# Build networkx graph from edges list:
graph = nx.read_edgelist("./data/graph1.txt", delimiter=",", create_using=nx.DiGraph,
                         data=(("label", int), ("attribute1", float,)))
graph = nx.convert_node_labels_to_integers(graph)

# Build the classifier.
classifier = EdgeClassifier("./pkl", "./plots", verbose=True, gpu=False)

# Define parameters to the graph and to the model and execute.
classifier.build("graph5", graph, {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 150
}, topological_features=None, data_features=["attribute1"])

```



[//]: # (This package classify graphs' edges by , graph edges classification by topological attributes)

[//]: # (> Attention! This package uses non boost graph-measures, and that's might make the features calculation slower. )
[//]: # (> If you would like to clac them in boost environment, follow the instruction here &#40;link&#41;, and move th .pkl file to the pkl directory.)