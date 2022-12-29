import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from EdgeClassifier.params_model import Model
from EdgeClassifier.mydataset import MyDataset


class Activator:
    def __init__(self, x_train, y_train, x_test, y_test, plots_dir, is_nni=False, gpu=False,
                 params={}, nni_metric='loss', should_plot=True, gpu_device=0):
        self.is_nni = is_nni
        self.verbose = True
        self.should_tqdm = True
        self.plots_dir = plots_dir
        self.should_plot = should_plot
        self.gpu = gpu
        self.gpu_device = gpu_device

        if self.is_nni:
            import nni
            params = nni.get_next_parameter()
            self.nni_metric = nni_metric

        self.lr = 5e-4 if 'lr' not in params else params['lr']
        weight_decay = 0 if 'l2' not in params else params['l2']
        dropout = 0.2 if 'dropout' not in params else params['dropout']
        self.epochs = 100 if 'epochs' not in params else params['epochs']
        train_batch_size = 512 if 'batch_size' not in params else params['batch_size']
        self.loss_func = torch.nn.BCELoss() if 'loss' not in params else params['loss']
        self.num_classes = 2 if 'num_classes' not in params else params['num_classes']
        self.is_binary = self.num_classes == 2
        activation = torch.nn.Sigmoid() if 'activation' not in params else params['activation']

        in_size = len(x_train[0])  # to decide the size of the first layer in the model.
        layers = [(in_size, 250, True),
                  (250, 125, True),
                  (125, 1 if self.is_binary else self.num_classes, False)
                  ] if 'layers' not in params else params['layers']

        # Just to make sure that the inputs size fits to the number of features.
        layers[0] = (in_size, layers[0][1], layers[0][2])
        self._model = Model({
            "activation": activation,
            "dropout": dropout,
            "layers": layers,
            # "end_function": torch.nn.Softmax() if self.is_binary else lambda x: x
            "end_function": torch.nn.Sigmoid() if self.is_binary else torch.nn.LogSoftmax(dim=1)
        })
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=weight_decay)

        # initialize validation set from train
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

        # Z-scoring the data. The zscore parameters are set only according to the train features,
        # so the validation and the test won't affect the model.
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = torch.tensor(scaler.transform(x_train), dtype=torch.float)
        x_valid = torch.tensor(scaler.transform(x_valid), dtype=torch.float)
        x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float)

        # Copy the tensors to the CPU/GPU
        self.device = torch.device(f"cuda:{self.gpu_device}" if gpu and torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        x_train = x_train.to(self.device)
        y_train = y_train.to(torch.float).to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(torch.float).to(self.device)
        x_valid = x_valid.to(self.device)
        y_valid = y_valid.to(torch.float).to(self.device)

        self.train_loader = DataLoader(MyDataset(x_train, y_train), shuffle=True, batch_size=train_batch_size)
        self.validation_loader = DataLoader(MyDataset(x_valid, y_valid), shuffle=False, batch_size=len(y_valid))
        self.test_loader = DataLoader(MyDataset(x_test, y_test), shuffle=False, batch_size=len(y_test))

        # Lists for results figures
        self.train_losses = []
        self.train_auc = []
        self.train_acc = []

        self.valid_losses = []
        self.valid_auc = []
        self.valid_acc = []

        self.epochs_list = []

        # Used for early stopping
        self.valid_bad_epoch_counter = 0
        self.last_loss = 1e6  # bigger than any other loss

    def print_epoch_results(self, epoch, train_results, valid_results):
        msg = f"\n epoch {epoch} - " \
              f"[Train] Acc: {train_results['acc']:.3f} | " \
              f"Loss: {train_results['loss']:.5f} | "
        if self.is_binary:
            msg += f"Auc: {train_results['auc']:.3f} || "

        msg += f"[Validation] Acc: {valid_results['acc']:.3f} | " \
               f"Loss: {valid_results['loss']:.5f} | "
        if self.is_binary:
            msg += f"Auc: {valid_results['auc']:.3f}"

        msg += "\n"
        print(msg)

    # def build_data_loaders(self):
    def train(self, verbose=True, should_tqdm=True):
        self.verbose = verbose
        self.should_tqdm = should_tqdm

        for epoch in range(self.epochs):
            self.epochs_list.append(epoch)  # Because early stopping might happen.

            train_results = self.run_epoch(self.train_loader, is_training=True)
            with torch.no_grad():
                valid_results = self.run_epoch(self.validation_loader, is_training=False)

            if not self.is_nni and self.verbose:
                self.print_epoch_results(epoch, train_results, valid_results)

            if not self.is_nni and self.should_plot:
                self.train_losses.append(train_results['loss'])
                self.train_auc.append(train_results['auc'])
                self.train_acc.append(train_results['acc'])

                self.valid_losses.append(valid_results['loss'])
                self.valid_auc.append(valid_results['auc'])
                self.valid_acc.append(valid_results['acc'])

            # early stopping checking
            if valid_results['loss'] > self.last_loss:
                self.valid_bad_epoch_counter += 1
            else:
                self.valid_bad_epoch_counter = 0

            self.last_loss = valid_results['loss']
            if self.valid_bad_epoch_counter >= 10:
                if self.verbose:
                    print('early stopping after 10 bad epochs.')
                if self.is_nni:
                    nni.report_final_result(valid_results[self.nni_metric])
                break

            # nni report results:
            if self.is_nni:
                if epoch == self.epochs - 1:
                    nni.report_final_result(valid_results[self.nni_metric])
                elif epoch % 5 == 0:
                    nni.report_intermediate_result(valid_results[self.nni_metric])

        if not self.is_nni and self.verbose:
            # test dataset and figure creation
            test_results = self.run_epoch(self.test_loader, is_training=False)
            msg = f"\n[-- Test --] Acc: {test_results['acc']:.3f} | "
            if self.is_binary:
                msg += f"Auc: {test_results['auc']:.3f} | "
            msg += f"Loss: {test_results['loss']:.5f}\n"
            print(msg)

        if not self.is_nni and self.should_plot:
            if self.is_binary:
                self.plot_roc_curve(self.test_loader, os.path.join(self.plots_dir, "roc.png"))
            self.create_plots()

    def run_epoch(self, loader, is_training=True):
        """
            This function gets a dataloader, run its samples on the model and return its results.
            In case this is a training epoch, it also calls the optimizer and change the weight.
            :param loader: A dataloader with samples to run
            :param is_training: A boolean value tells if we are in train epoch.
            :return: A dictionary with the results of the running (acc, loss and auc)
        """
        # self._model.train() if is_training else self._model.eval()
        if is_training:
            self._model.train()
        else:
            self._model.eval()

        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        for features, labels in tqdm(loader, disable=(not self.should_tqdm) or (not is_training)):
            if is_training:
                self.optimizer.zero_grad()

            y_pred = self._model(features)
            if self.is_binary:
                loss = self.loss_func(y_pred, labels.to(torch.float))
            else:
                loss = self.loss_func(y_pred, labels.type(torch.long))

            if is_training:
                loss.backward()
                self.optimizer.step()

            # Keeps this data for the result in the end of the epoch
            epoch_preds.append(y_pred)  # floating point predications for all dataset
            epoch_labels.append(labels)
            epoch_loss += loss.item()  # sum avg loss of all batches

        # Combines the data from every batch to one tensor - for the results' calculation.
        labels = torch.concat(epoch_labels).cpu().detach().numpy()
        preds = torch.concat(epoch_preds).cpu().detach().numpy()

        return {
            "acc": accuracy_score(labels, preds > 0.5) if self.is_binary else
            accuracy_score(labels, np.argmax(preds, axis=1)),
            "auc": roc_auc_score(labels, preds) if self.is_binary else 0,
            "loss": epoch_loss / len(loader)
        }

    def create_plots(self):
        """
            This function creates and saves 3 figures which summary the epoch's results.
            Figures for accuracy, loss and roc auc, of the train and the validation.
        """
        # Accuracy plot
        plt.clf()
        plt.plot(self.epochs_list, self.train_acc, label="acc train")
        plt.plot(self.epochs_list, self.valid_acc, label="acc validation")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "acc.png"))

        if self.is_binary:
            # Roc auc plot
            plt.clf()
            plt.plot(self.epochs_list, self.train_auc, label="auc train")
            plt.plot(self.epochs_list, self.valid_auc, label="auc validation")
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "auc.png"))

        # Loss plot
        plt.clf()
        plt.plot(self.epochs_list, self.train_losses, label="loss train")
        plt.plot(self.epochs_list, self.valid_losses, label="loss valid")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "loss.png"))

    def plot_roc_curve(self, loader, path):
        """
            This function run the loader's samples on the model and create the roc curve.
            :param loader: The loader of the samples to test
            :param path: The path to put the roc figure in.
        """
        # Try the samples onm the model
        epoch_preds = []
        epoch_labels = []
        for features, labels in loader:
            y_pred = self._model(features)

            epoch_preds.append(y_pred)  # floating point predications for all dataset
            epoch_labels.append(labels)

        labels = torch.concat(epoch_labels).cpu().detach().numpy()
        preds = torch.concat(epoch_preds).cpu().detach().numpy()

        # Builds and saves the roc curve
        plt.clf()
        fpr, tpr, thresholds = roc_curve(labels, preds)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(path)

    @property
    def model(self):
        return self._model
