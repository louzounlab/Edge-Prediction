import os
import networkx as nx
from EdgeClassifier.activator import Activator
from EdgeClassifier.features_builder import FeaturesBuilder


class EdgeClassifier:
    def __init__(self, pkl_folder: str, plots_folder: str, verbose=True, should_tqdm=True, gpu=False, should_plot=True):
        self.activator = None
        self.builder = None
        self.pkl_folder = pkl_folder
        self.plots_folder = plots_folder
        self.gpu = gpu
        self.verbose = verbose
        self.should_plot = should_plot
        self.should_tqdm = should_tqdm

    def build(self, graph_name: str, graph: nx.Graph, params={}, topological_features=None, data_features=[],
              force_build=False):

        v1, v2 = list(graph.edges)[0]
        if 'label' not in graph[v1][v2]:
            raise AttributeError("Edges should have a label attribute.")

        # Creates the folders for the plots of this graph
        plots_path = os.path.join(self.plots_folder, graph_name)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # Builds the features of the edges
        builder = self.builder = FeaturesBuilder(graph_name, graph, self.pkl_folder, test_size=0.15,
                                                 topology_features=topological_features, data_features=data_features)
        builder.build(force_build=force_build)

        # Train the model
        x_train, y_train, x_test, y_test = builder.x_train, builder.y_train, builder.x_test, builder.y_test

        activator = self.activator = Activator(x_train, y_train, x_test, y_test, plots_path, gpu=self.gpu,
                                               params=params, should_plot=self.should_plot)
        activator.train(should_tqdm=self.should_tqdm)

    @property
    def model(self):
        return self.activator.model

    @property
    def data_train(self):
        return self.builder.x_train, self.builder.y_train

    @property
    def data_test(self):
        return self.builder.x_test, self.builder.y_test


"""
Things i would like to add:
* An option to decide about the model's layers - there is a code but it doesn't work - ask Amit!
* README.md
* Option to use multiclass instead binary (loss, labeling, out_size, softmax, auc, etc.)
* loss = user can choose. labeling = we should use argmax?, auc = cancel not in binary, out_size and softmax is in params model. - ask Amit!  
"""
