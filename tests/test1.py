import sys
# temp until i will figure how to solve it
sys.path.append("..")

import networkx as nx
from EdgeClassifier.edge_classifier import EdgeClassifier
# from EdgeClassifier import EdgeClassifier # why this isn't working even with the init file?

if __name__ == '__main__':
    # with open()
    graph = nx.read_edgelist("./data/graph1.txt", delimiter=",", create_using=nx.DiGraph,
                             data=(("label", int), ("f1", int),))
    graph = nx.convert_node_labels_to_integers(graph)

    classifier = EdgeClassifier("./pkl", "./plots", verbose=True, gpu=False)
    classifier.build("graph5", graph, {
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 150
    }, topological_features=None, data_features=["f1"])
