####################################################################################################
# Name         : ml_models.py
# Description  : This file contains function to train and test ML models for titanic classification
#                dataset.
# Language     : Python 3.6
# Requirements : tqdm, scikit-learn, xgboost, lightgbm
####################################################################################################

# Import required packages
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .utils import calculate_class_weight, print_kfold_data

def train_ml_models(features, labels, models="all", n_splits=4):

    """
    Function to train ML models with stratified K-fold.

    Arguments:
        features : (numpy.ndarray) The input features of the data.
        labels   : (numpy.ndarray) The target values of the data.
        models   : (list or "all") The list of models to train. If "all" is given, all models will
                   be trained.
        n_splits : (int) Number of splits for stratified K-fold.

    Return:
        dict : The training results
               {
                   <model_name> : {
                       Fold_1 : {
                           Accuacy score : <float>
                           F1 score : <float>
                           F1 score per class : <float>
                           model : <scikit-learn model>
                       }
                       Fold_n : {
                           Accuacy score : <float>
                           F1 score : <float>
                           F1 score per class : <float>
                           model : <scikit-learn model>
                       }
                   }
               }
    """

    # Calculate class weights
    classes, _, class_weights_dict = calculate_class_weight(labels)

    # Definitionf for all models
    model_dict = {
        "LogisticRegression" : LogisticRegression(class_weight=class_weights_dict),
        "GaussianNB" : GaussianNB(),
        "BernouliNB" : BernoulliNB(),
        "KNeighborsClassifier" : KNeighborsClassifier(),
        "DecisionTreeClassifier" : DecisionTreeClassifier(class_weight=class_weights_dict),
        "BaggingClassifier" : BaggingClassifier(),
        "RandomForestClassifier" : RandomForestClassifier(class_weight=class_weights_dict),
        "AdaBoostClassifier" : AdaBoostClassifier(),
        "LGBMClassifier" : LGBMClassifier(class_weight=class_weights_dict),
        "XGBClassifier" : XGBClassifier(),
        "ExtraTreesClassifier" : ExtraTreesClassifier(class_weight=class_weights_dict),
        "LinearSVC" : SVC(kernel="linear", class_weight=class_weights_dict),
        "SVC" : SVC(class_weight=class_weights_dict),
        "PolySVC" : SVC(kernel="poly", class_weight=class_weights_dict),
        "SGDClassifier" : SGDClassifier(class_weight=class_weights_dict),
        "Lasso" : Lasso(),
        "RidgeClassifier" : RidgeClassifier(class_weight=class_weights_dict),
        "NearestCentroid" : NearestCentroid(),
        "MLPClassifier" : MLPClassifier(hidden_layer_sizes=(64,64,64), max_iter=500),
    }

    # Identify the required models
    req_models = []
    if models == "all":
        req_models = list(model_dict.keys())

    elif type(models) == list:
        for cur_model in models:
            if cur_model in model_dict:
                req_models.append(cur_model)
            else:
                print("Model {} is not found. Skipping {}".format(cur_model, cur_model))

    # If valid models are found in required models
    if len(req_models) > 0:

        training_results = {}
        kfold_details = {}

        # Perform stratified k-fold split on the data
        skf = StratifiedKFold(n_splits=n_splits)
        for model_no, (train_index, valid_index) in enumerate(skf.split(features, labels)):
            train_features, valid_features = features[train_index], features[valid_index]
            train_labels, valid_labels = labels[train_index], labels[valid_index]

            # Store the details of the train and test data
            kfold_details["Fold_" + str(model_no+1)] = {
                    "label count" : {idx : list(train_labels).count(idx) for idx in classes},
                    "Training samples" : len(train_labels),
                    "Validation samples" : len(valid_labels)
            }

            # Iterate through the required models
            pbar = tqdm(req_models)
            for cur_model in pbar:

                pbar.set_description("Fold : {}/{}".format(model_no+1, n_splits))

                # Train the model
                model_dict[cur_model].fit(train_features, train_labels)

                # Predict using the trained model
                predictions = model_dict[cur_model].predict(valid_features)

                # Lasso will give regressed output between -1 and 1
                if cur_model == "Lasso":
                    predictions = [1 if pred > 0 else 0 for pred in predictions]

                # Calculate the metrics
                accuracy = accuracy_score(valid_labels, predictions)
                f1 = f1_score(valid_labels, predictions, average="macro")
                f1_per_class = {}
                for idx, score in enumerate(f1_score(valid_labels, predictions, average=None)):
                    f1_per_class[idx] = score

                if cur_model not in training_results:
                    training_results[cur_model] = {}

                # Store the results
                training_results[cur_model]["Fold_" + str(model_no+1)] = {
                    "Accuracy score" : accuracy,
                    "F1 score" : f1,
                    "F1 score per class" : f1_per_class,
                    "model" : model_dict[cur_model],
                }

    # When no valid models found in required models
    else:
        training_results = "No valid model names found. Model names should be from {}".format(list(model_dict.keys()))

    # Display the K-Fold split details
    print_kfold_data(kfold_details)

    return training_results