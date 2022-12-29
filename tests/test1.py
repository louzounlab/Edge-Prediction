import sys
# temp until I will figure how to solve it
# sys.path.append("..")

import networkx as nx

from EdgeClassifier import EdgeClassifier

if __name__ == '__main__':
    # with open()
    graph = nx.read_edgelist("./data/new.txt", delimiter=",", create_using=nx.DiGraph,
                             data=(("label", int),))
                             # data=(("label", int), ("f1", int),))

    classifier = EdgeClassifier("./pkl", "./plots", verbose=True, gpu=False)
    classifier.build("cora", graph, {
        "lr": 0.0001,
        "epochs": 200,
        "num_classes": 2,
        "dropout": 0
    }, topological_features=None, data_features=[])
