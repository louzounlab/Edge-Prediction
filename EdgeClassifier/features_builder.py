import os.path as path
import pickle

import networkx as nx
import torch
from graphMeasures import FeatureCalculator
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FeaturesBuilder:
    """
        This class gets a networkx graph, and calculates topological attributes for each edge in the graph.
        The attributes of each edge and it's label are saved in tensors, which devided to train and test.
        There for, a classification algorithm can use easily x_train, y_train, x_test, y_test for edge label
        prediction task.
    """

    def __init__(self, graph_name: str, nxg, pkl_folder, test_size=0.25, topology_features=None, data_features=[]):
        """
        Initializes the class with the graph and other necessary information.
        :param graph_name: The name of the graph (for pickling)
        :param nxg: A networkx graph with *data named label*
        :param pkl_folder: Path to a folder which the pickle files will be saved inside
        :param test_size: Size of the test group from all the edges (between 0 and 1)
        """
        if data_features is None:
            data_features = []
        self.pkl_folder = pkl_folder
        self.graph_pkl_path = path.join(pkl_folder, graph_name)

        self.graph_name = graph_name
        # This is an important line -
        # it names the nodes by the order they will be ordered in the output of graph-measure.
        self.nxg = nx.convert_node_labels_to_integers(nxg)

        self._nodes_features = None
        self._edges_features = None
        self._labels = None
        self.number_of_edge_features = None
        self.number_of_node_features = None

        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None

        self.test_size = test_size
        self.features_list = topology_features if topology_features is not None else \
            ["in_degree", "out_degree", "closeness_centrality", "bfs_moments", "page_rank", "motif3"]
        self.data_features = data_features

    def pickle_load(self, pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def pickle_dump(self, pkl_path, obj):
        with open(pkl_path, "wb") as f:
            pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)

    def build(self, force_build=False):
        pkl_path_x_train = path.join(self.graph_pkl_path, "x_train.pkl")
        pkl_path_x_test = path.join(self.graph_pkl_path, "x_test.pkl")
        pkl_path_y_train = path.join(self.graph_pkl_path, "y_train.pkl")
        pkl_path_y_test = path.join(self.graph_pkl_path, "y_test.pkl")

        if path.exists(pkl_path_x_train) and path.exists(pkl_path_x_test) and \
                path.exists(pkl_path_y_train) and path.exists(pkl_path_y_test) and not force_build:
            self.x_train = self.pickle_load(pkl_path_x_train)
            self.x_test = self.pickle_load(pkl_path_x_test)
            self.y_train = self.pickle_load(pkl_path_y_train)
            self.y_test = self.pickle_load(pkl_path_y_test)

        else:
            # Calculates the edge's feature
            self.calc_nodes_features()
            self.calc_edge_features()
            self.apply_log()  # the log doesn't have any parameter, so it's ok to apply it on all data

            # Split the edge's features and labels into train and test.
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self._edges_features, self._labels, test_size=self.test_size)

            # Keeps the train/test objects in pkl folder for next use.
            self.pickle_dump(pkl_path_x_train, self.x_train)
            self.pickle_dump(pkl_path_x_test, self.x_test)
            self.pickle_dump(pkl_path_y_train, self.y_train)
            self.pickle_dump(pkl_path_y_test, self.y_test)

    def calc_nodes_features(self):
        # FeatureCalculator Verbose=False has a bug. When it will be fixed, we will give the option to shut it down.
        ftr_calc = FeatureCalculator(self.nxg.copy(), self.features_list, dir_path=self.graph_pkl_path, acc=False,
                                     directed=True, gpu=False, device=0, verbose=True, should_zscore=False)
        ftr_calc.calculate_features()
        self._nodes_features = ftr_calc.feature_matrix

    def set_topological_features(self, source, target, edge_num):
        """
        :param source: The index of the source node of the edge
        :param target: The index of the target node of the edge
        :param edge_num: The number of the edge in the list
        :param num: Number of features to clac avg and sub from them.
        """
        for index in range(self.number_of_node_features):
            # For the index node's feature - add the subtraction and the average.
            self._edges_features[edge_num][2 * index] = \
                (self._nodes_features[source][index] - self._nodes_features[target][index]) / 2
            self._edges_features[edge_num][2 * index + 1] = \
                (self._nodes_features[source][index] + self._nodes_features[target][index]) / 2

    def set_data_features(self, edge_num, dic: dict):
        base = self.number_of_node_features * 2
        for i, key in enumerate(self.data_features):
            self._edges_features[edge_num][base + i] = dic[key]

    def calc_edge_features(self):
        """
            Calculates the sizes and create tensors for the edges-features and the labels.
            Then, create two edge features from each node feature (avg and sub of the 2 node's feature)
        """
        self.number_of_node_features = self._nodes_features.shape[1]
        self.number_of_edge_features = 2 * self.number_of_node_features + len(self.data_features)

        # Creates torch with size - number of edges X number of features - which will keep the edges features
        self._edges_features = torch.empty((self.nxg.number_of_edges(), self.number_of_edge_features),
                                           dtype=torch.float)
        self._labels = torch.empty(self.nxg.number_of_edges(), dtype=torch.int8)

        # For each edge - calculates the features from the node features of the edge's nodes.
        for edge_index, (source, target, dic) in tqdm(enumerate(self.nxg.edges(data=True))):
            self.set_topological_features(source, target, edge_index)
            self.set_data_features(edge_index, dic)
            self._labels[edge_index] = dic['label']  # The multiclass case

    def apply_log(self):
        """
            The function takes every second column (columns which are an average of other non-negative features),
            adds them 0.01 (to make them positive for sure) and then apply 10 base log on them.
        """
        for i in range(self.number_of_node_features):
            # apply log(x + 0.01) only on sure positive attributes
            self._edges_features[:, 2 * i + 1].add_(0.01)
            self._edges_features[:, 2 * i + 1].log10_()
