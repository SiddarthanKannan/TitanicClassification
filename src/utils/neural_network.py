####################################################################################################
# Name         : neural network.py
# Description  : This file contains functions to train and test custom nerual network for titanic
#                classification dataset.
# Language     : Python 3.6
# Requirements : torch, scikit-learn, tqdm
####################################################################################################

# Import necessary packages
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .utils import calculate_class_weight

class CustomNetwork(torch.nn.Module):
    """
    Custom Neural Network classifier.
    """

    def __init__(self, n_input, n_layers, n_hidden, dropout=0.2):
        """
        Function to initialize the model.

        Arguments:
            n_input  : (int) Number of input features.
            n_layers : (int) Number of hidden layers.
            n_hidden : (int) Number of hidden layer neurons.
            dropout  : (float) Droput ratio. default is 0.2.

        Return:
            None
        """
        super().__init__()

        # Intialize first layer
        self.hidden_layer1 = torch.nn.Linear(n_input, n_hidden)
        self.act1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout)

        # Initalize other hidden layers
        self.hidden_layers = []
        for _ in range(1, n_layers):
            self.hidden_layers.append([
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])

        # Initalize the last layer
        self.output = torch.nn.Linear(n_hidden, 2)

    def forward(self, input_data):
        """
        This function passes the data through the network and returns the output

        Arguments:
            input_data : (torch.tensor) Input data (n_sample * n_features)

        Return:
            torch.tensor : Ouput from the model (n_sample * 2)

        """

        # Pass through initail layers
        data = self.hidden_layer1(input_data)
        data = self.act1(data)
        data = self.drop1(data)

        # Pass through other hidden layers
        for layer in self.hidden_layers:
            data = layer[2](layer[1](layer[0](data)))

        # Pass through the final layer
        data = self.output(data)

        return data

class CustomNetTrainer:
    """
    Class to train and test the custom neural network.
    """

    def __init__(self, learning_rate=0.001):
        """
        Initialize the training parameters.

        Arguments:
            learning_rate : (float) Learning rate for the training. Default is 0.001.

        Return:
            None
        """

        self.learning_rate = learning_rate

    def train(self, features, labels, n_epochs, n_splits=4):
        """
        Function to train custom neural network using K-fold on the given data.

        Arguments:
            features : (numpy.ndarray) The input data to train.
            labels   : (numpy.ndarray) The target variable of the data to train.
            n_epochs : (int) The number of epochs to train.
            n_splits : (int) Number of splits for k-fold.

        Return:
            (dict, dict)

            dict : The metrics in each epoch.
                    {
                        Fold_1 : {
                            "train loss" : <list>,
                            "valid loss" : <list>,
                            "train accuracy" :<list>,
                            "valid accuracy" : <list>,
                            "train f1" : <list>,
                            "valid f1" : <list>
                        }
                        Fold_n : {
                            "train loss" : <list>,
                            "valid loss" : <list>,
                            "train accuracy" :<list>,
                            "valid accuracy" : <list>,
                            "train f1" : <list>,
                            "valid f1" : <list>
                        }
                    }

            dict : The training results.
                    {
                        Fold_1 : {
                            "Accuracy score" : <float>,
                            "F1 score" : <float>,
                            "F1 score per class" : <dict>,
                            "model" : <str>
                        }
                        Fold_n : {
                            "Accuracy score" : <float>,
                            "F1 score" : <float>,
                            "F1 score per class" : <dict>,
                            "model" : <str>
                        }
                    }

        """

        # Calculate class weight
        _, class_weight, _ = calculate_class_weight(labels)

        # Initialize loss function
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float))

        training_results = {}
        progress_results = {}

        # Perform stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits)
        for model_no, (train_index, valid_index) in enumerate(skf.split(features, labels)):
            train_features, valid_features = features[train_index], features[valid_index]
            train_labels, valid_labels = labels[train_index], labels[valid_index]

            # Initialize model and optimizer
            model = CustomNetwork(features.shape[1], 3, 64)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            train_losses = []
            test_losses = []
            train_accs = []
            test_accs = []
            train_f1s = []
            test_f1s = []
            train_f1s_per_class = []
            test_f1s_per_class = []

            pbar = tqdm(range(1, n_epochs+1))
            for _ in pbar:

                pbar.set_description("Fold : {}/{}".format(model_no+1, n_splits))

                # Train the model
                model.train()

                out = model(torch.tensor(train_features, dtype=torch.float))

                # Store the loss and metrics
                train_loss = criterion(out, torch.tensor(train_labels))
                train_acc = accuracy_score(torch.argmax(out, dim=1), train_labels)
                train_f1 = f1_score(torch.argmax(out, dim=1), train_labels, average="macro")
                train_f1_per_class = f1_score(torch.argmax(out, dim=1), train_labels, average=None)

                # Update the model parameters
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Validate the model
                out = self.predict(model, valid_features)

                # Store the validation loss and metrics
                test_loss = criterion(out, torch.tensor(valid_labels))
                test_acc = accuracy_score(torch.argmax(out, dim=1), valid_labels)
                test_f1 = f1_score(torch.argmax(out, dim=1), valid_labels, average="macro")
                test_f1_per_class = f1_score(torch.argmax(out, dim=1), valid_labels, average=None)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                train_f1s.append(train_f1)
                test_f1s.append(test_f1)
                train_f1s_per_class.append(train_f1_per_class)
                test_f1s_per_class.append(test_f1_per_class)

            # Store the progress
            progress_results["Fold_" + str(model_no+1)] = {
                "train loss" : train_losses,
                "valid loss" : test_losses,
                "train accuracy" :train_accs,
                "valid accuracy" : test_accs,
                "train f1" : train_f1s,
                "valid f1" : test_f1s
            }

            # Store the training results
            training_results["Fold_" + str(model_no+1)] = {
                "Accuracy score" : test_accs[-1],
                "F1 score" : test_f1s[-1],
                "F1 score per class" : {idx:score for idx, score in enumerate(test_f1s_per_class[-1])},
                "model" : model
            }

        return progress_results, training_results

    def predict(self, model, features):
        """
        This function is used to predict using the given custom neural network.

        Arguments:
            model    : (CustomNetwork) The model used to predict.
            features : (numpy.ndarray) The input data to predict.

        Return:
            (torch.tensor) : Ouput from the model.
        """

        model.eval()
        with torch.no_grad():

            out = model(torch.tensor(features, dtype=torch.float))
            return out


