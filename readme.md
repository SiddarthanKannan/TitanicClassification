# Titanic Data Classification
---

## Table of contents

- [Description](#description)
- [Data source](#data-source)
- [Execution](#Execution)
- [Results on test data](#results-on-test-data)
- [Kaggle public score](#kaggle-public-score)


### Description

The objective of this task is to predict whether a passenger survived the Titanic distaster based on the given data.

### Data source

The data used for this task can be downloaded from the kaggle [website](https://www.kaggle.com/c/titanic/data)

### Execution

There are three ipython notebooks that needs to be executed

1. [data_analysis.ipynb](src/data_analysis.ipynb) - This file contains code to analysis the titanic dataset.
2. [data_preprocessing.ipynb](src/data_preprocessing.ipynb) - This file preprocess the data based on the findings from the analysis.
3. [train_and_test.ipynb](src/train_and_test.ipynb) - This file trains the processed data on various models and tests using best models.

### Results on test data

Validation results of top models

|Models|Fold 1 F1 score|Fold 2 F1 score|Fold 3 F1 score|Fold 4 F1 score|Fold 1 Accuracy|Fold 2 Accuracy|Fold 3 Accuracy|Fold 4 Accuracy|
|---|---|---|---|---|---|---|---|---|
|LGBMClassifier	        |0.771867	|0.821675	|0.822042	|0.805857   |0.784753	|0.829596	|0.834081	|0.815315|
|AdaBoostClassifier     |0.771368   |0.821675   |0.778833   |0.840860   |0.780269   |0.829596   |0.798206   |0.851351|
|SVC                    |0.783257	|0.814352	|0.778833	|0.829467   |0.793722	|0.820628	|0.798206	|0.842342|
|XGBClassifier	        |0.765480	|0.834024	|0.816770	|0.789337   |0.780269	|0.843049	|0.829596	|0.801802|
|MLPClassifier	        |0.762361	|0.828109	|0.777039	|0.831215   |0.775785	|0.838565	|0.793722	|0.842342|
|RandomForestClassifier |0.739515   |0.828900   |0.795323   |0.793640   |0.762332   |0.838565   |0.811659   |0.806306|

### Kaggle public score

Public scores of the predictions using the top models on Kaggle.

|Models                 |Accuracy|
|-----------------------|--------|
|LGBMClassifier	        |0.73205 |
|AdaBoostClassifier     |0.75837 |
|SVC                    |0.75837 |
|XGBClassifier	        |0.75358 |
|MLPClassifier	        |0.75837 |
|RandomForestClassifier |0.74401 |





