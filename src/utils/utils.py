####################################################################################################
# Name         : utils.py
# Description  : This file contains utility functions to train and test ML models for titanic
#                classification dataset.
# Language     : Python 3.6
# Requirements : numpy, pandas, matplotlib, scikit-learn, prettytable
####################################################################################################

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from prettytable import PrettyTable


def calculate_class_weight(labels):
    """
    This function is used to calculate the class weights to overcome class imbalance

    Arguments:
        labels : (numpy.ndarray) The target values in the data.

    Returns:
        (numpy.ndarray, list, dict)

        numpy.ndarray : The unique classes in the target.
        list          : The class weights list in the order of classes as in above array.
        dict          : Dictionary of class weights.
    """
    # Calculate class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)

    # Store class weights as dict
    class_weights_dict = {y:class_weights[y] for y in np.unique(labels)}

    return np.unique(labels), class_weights, class_weights_dict

def get_results_data(output):
    """
    Function to convert the training result dictionary to dataframe.

    Arguments:
        output : (dict) The results of ML model training.

    Return:
        pandas.DataFrame : The results in dataframe.
    """


    # List of model names
    model_names = list(output.keys())
    results = []

    # iterate through the dictionary
    for model_name in output:
        cur_results = []
        columns = []
        n_splits = len(output[model_name])

        # Get the metrics from the dictionary
        for metric in ["Accuracy score", "F1 score"]:
            for fold in output[model_name]:
                cur_results.append(
                    output[model_name][fold][metric],
                )
                columns.append(metric + " " + fold)
            cur_results.append(np.average(cur_results[-n_splits:]))
            columns.append(metric + " average")

        # Get the F1 score per class from the dictionary
        for class_idx in output[model_name][fold]["F1 score per class"]:
            for fold in output[model_name]:
                cur_results.append(
                    output[model_name][fold]["F1 score per class"][class_idx],
                )
                columns.append("F1 score for class " + str(class_idx) + " " + fold)
            cur_results.append(np.average(cur_results[-n_splits:]))
            columns.append("F1 score for class " + str(class_idx) + " average")

        results.append(cur_results)

    # Store the data in dataframe
    result_df = pd.DataFrame(results, columns=columns, index=model_names)

    return result_df

def get_best_model(result_df):
    """
    This function identifies best models based on accuracy and F1 score

    Arguments:
        result_df : (pandas.DataFrame) The output from get_results_data function

    Returns:
        list : List of best models
    """

    # Function to calculate the difference between max and min values in list
    def get_diff(data):
        return max(data) - min(data)

    # Identify the accuracy and f1 score columns
    acc_columns = [col for col in result_df.columns if "Accuracy score Fold" in col]
    acc_avg_column = "Accuracy score average"
    f1_columns = [col for col in result_df.columns if "F1 score Fold" in col]
    f1_avg_column = "F1 score average"

    # Identify the best models
    best_models = []
    for model_name in result_df.index:
        data = result_df.loc[model_name]

        # Condition for best model (K-fold metrics do not differ by more than 10% and
        # either accuracy or F1 score should be greater than 80%)
        if get_diff(data[acc_columns]) < 0.1 and get_diff(data[f1_columns]) < 0.1 and \
            (data[acc_avg_column] >= 0.8 or data[f1_avg_column] >= 0.8):
            best_models.append(model_name)

    return best_models

def print_kfold_data(kfold_details):
    """
    This function prints the sample information for each fold in k-fold split.

    Arguments:
        kfold_details : (dict) The kfold split data from ML training.

    Return:
        None
    """

    # Get the target variables
    labels = list(kfold_details[list(kfold_details.keys())[0]]["label count"].keys())

    # Intialize pretty table to print
    table = PrettyTable()

    # Headers for the table
    table.field_names = ["Fold", "Training samples", "Validation samples"] + \
                        ["Samples for class " + str(label) for label in labels]

    # Get the details for each split
    for fold in kfold_details:

        train_samples = kfold_details[fold]["Training samples"]
        valid_samples = kfold_details[fold]["Validation samples"]
        cur_data = [fold, train_samples, valid_samples]

        for label in kfold_details[fold]["label count"]:
            cur_data.append(kfold_details[fold]["label count"][label])

        table.add_row(cur_data)

    # Print the details
    print(table)



def plot_progress(progress_results):
    """
    This function is used to plot the neural network training progress

    Arguments:
        progress_results : (dict) The dictionary containing loss and metrics for each epoch.

    Return:
        list : List of matplotlib figures for each split in k-fold.
    """

    # List to store the figures
    all_figs = []

    # Get the data from the dictionary
    for fold in progress_results:
        train_loss = progress_results[fold]["train loss"]
        valid_loss = progress_results[fold]["valid loss"]
        train_accuracy = progress_results[fold]["train accuracy"]
        valid_accuracy = progress_results[fold]["valid accuracy"]
        train_f1 = progress_results[fold]["train f1"]
        valid_f1 = progress_results[fold]["valid f1"]

        epochs = list(range(1, len(train_loss)+1))

        # Plot loss graph
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_loss, label="training")
        plt.plot(epochs, valid_loss, label="validation")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Epochs vs Loss")

        # Plot accuracy graph
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accuracy, label="training")
        plt.plot(epochs, valid_accuracy, label="validation")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("Epochs vs Accuracy")

        # Plot F1 score graph
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_f1, label="training")
        plt.plot(epochs, valid_f1, label="validation")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("f1")
        plt.title("Epochs vs F1")

        plt.suptitle(fold)

        # Store the figure
        all_figs.append(fig)

    return all_figs