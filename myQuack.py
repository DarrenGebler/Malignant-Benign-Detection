'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import csv


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''

    return [(9941835, 'Darren', 'Gebler'), (9983601, 'Davide', 'Dolcetti')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    """
    Read a comma separated text file where
    - the first field is a ID number
    - the second field is a class label 'B' or 'M'
    - the remaining fields are real-valued

    Return two numpy arrays X and y where
    - X is two dimensional. X[i,:] is the ith example
    - y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
    X,y
    """
    data_list = list(csv.reader(open(dataset_path), delimiter=','))
    X = np.array(data_list)
    y = [val[1] for val in X]
    for n, i in enumerate(y):
        if i == 'M':
            y[n] = 1
        elif i == 'B':
            y[n] = 0
    X = X[:, 2:]
    y = np.array(y)
    return X.astype(np.float64), y.astype(np.int)






# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_DecisionTree_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """
    decision_tree_classifier = DecisionTreeClassifier(random_state=1)

    decision_tree_classifier.fit(X_training, y_training)

    return decision_tree_classifier


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    """
    Build a Nearest Neighbours classifier based on the training set X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """
    nearest_neighbour_classifier = NearestNeighbors()

    nearest_neighbour_classifier.fit(X_training, y_training)

    return nearest_neighbour_classifier



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """
    svm_classifier = SVC(random_state=1, gamma='auto')

    svm_classifier.fit(X_training, y_training)

    return svm_classifier


if __name__ == '__main__':
    path = "medical_records.data"
    X, y = prepare_dataset(path)
    X_training, X_val, y_training, y_val = train_test_split(X, y, random_state=0)

    decision_tree_classifier = build_DecisionTree_classifier(X_training, y_training)
    nearest_neighbour_classifier = build_NearrestNeighbours_classifier(X_training, y_training)
    svm_classifier = build_SupportVectorMachine_classifier(X_training, y_training)

    val_predictions_dt = decision_tree_classifier.predict(X_val)
    # val_predictions_nn = nearest_neighbour_classifier.score(X_val)
    val_predictions_svm = svm_classifier.predict(X_val)

    print("MAE for Decision Tree Classifier: {}".format(mean_absolute_error(y_val, val_predictions_dt)))
    # print("MAE for Nearest Neighbour Classifier: {}".format(mean_absolute_error(y_val, val_predictions_nn)))
    print("MAE for SVM Classifier: {}".format(mean_absolute_error(y_val, val_predictions_svm)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    """
    Build a Neural Network classifier (with two dense hidden layers)
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """

    neural_net_classifier = tf.keras.models.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# if __name__ == "__main__":
#     pass
#     # Write a main part that calls the different
#     # functions to perform the required tasks and repeat your experiments.
#     # Call your functions here
#
#     ##         "INSERT YOUR CODE HERE"
#     raise NotImplementedError()
