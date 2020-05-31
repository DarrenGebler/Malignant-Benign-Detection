'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

import os
import random
import numpy as np
seed_value = 56
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(seed_value)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    """

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

    :param dataset_path: full path of the dataset text file

    :return X,y
    """
    # Define array with the correspondent name of each value in the given data
    column_headers = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
                      'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry',
                      'Mean Fractal', 'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness',
                      'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal',
                      'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
                      'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal']

    # Read the data from file
    data = pd.read_csv(dataset_path, names=column_headers)

    # create numpy arrays
    X = data.iloc[:, 2:32]
    y = data.iloc[:, 1]

    # Label data and set 1 for "M" and 0 for "B"
    diagnosis_encoder = LabelEncoder()
    y = diagnosis_encoder.fit_transform(y)

    # Return numpy arrays
    return X, y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_DecisionTree_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training, y_training.

    :param X_training: X_training[i,:] is the ith example
    :param y_training: y_training[i] is the class label of X_training[i,:]

    :return the classifier built in this function
    """
    # Split Training Dataset into, Train and Validate Datasets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    # Define parameters to be tuned by GridSearchCV
    tuned_parameters = [{'criterion': ['gini'], 'max_depth': [1, 5, 25, 50, 100, 250, 500],
                         'max_leaf_nodes': [None, 5, 25, 50, 100, 250, 500], 'random_state': [1]}]

    print("# Tuning hyper-parameters for precision")
    print()

    # Find best parameters to use based on tuned_parameters. Score on precision
    dt_cv = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters, scoring='precision'
    )

    # Fit model to train data
    dt_cv.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(dt_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    # Print mean, standard deviation and parameters of each combination of parameters
    means = dt_cv.cv_results_['mean_test_score']
    stds = dt_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, dt_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # Print classification report using validation data
    y_true, y_pred = y_val, dt_cv.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    # Set Decision Tree Classifier model with best parameters
    decision_tree_classifier = DecisionTreeClassifier(criterion=dt_cv.best_params_['criterion'],
                                                      max_depth=dt_cv.best_params_['max_depth'],
                                                      max_leaf_nodes=dt_cv.best_params_['max_leaf_nodes'], random_state=1)

    # Train Decision Tree Classifier model with training dataset
    decision_tree_classifier.fit(X_training, y_training)

    # Return decision tree classifier model
    return decision_tree_classifier


def decision_tree_test(dt_model, X_test, y_test):
    """
    Predicts data based on a X_test dataset using Decision Tree model. Compares results with y_test data and returns
    an MAE and Accuracy Score, along with a confusion matrix and heatmap for visualisation.
    :param dt_model: A trained Decision Tree model. Trained in build_DecisionTree_classifier function
    :param X_test: X value test set
    :param y_test: y value test set

    """
    # Predict y_values of test set
    val_predictions_dt = dt_model.predict(X_test)

    print("MAE for Decision Tree Classifier: {}".format(mean_absolute_error(y_test, val_predictions_dt)))
    print("Accuracy Score for Decision Tree Classifier: {}".format(accuracy_score(y_test, val_predictions_dt)))

    # Create confusion matrix with right and wrong estimations
    cm = confusion_matrix(y_test, val_predictions_dt)

    # Create matplotlib figure
    dt_plot = plt.axes()

    # Use seaborn library to create heatmap
    sns.heatmap(cm, annot=True, fmt='d', ax=dt_plot)

    # Set figure title
    dt_plot.set_title("Decision Tree Heatmap")

    # Show figure
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    """
    Build a Nearest Neighbours classifier based on the training set X_training, y_training.

    :param X_training: X_training[i,:] is the ith example
    :param y_training: y_training[i] is the class label of X_training[i,:]

    :return the classifier built in this function
    """
    # Split Training Dataset into, Train and Validate Datasets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    # Define parameters to be tuned by GridSearchCV
    tuned_parameters = [
        {'n_neighbors': [1, 3, 5, 10, 15, 20, 30, 50, 100], 'leaf_size': [1, 3, 5, 10, 15, 20, 30, 50, 100]}]

    print("# Tuning hyper-parameters for %s" % 'precision')
    print()

    # Find best parameters to use based on tuned_parameters. Score on precision
    nn_cv = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring='precision')

    # Fit model to train data
    nn_cv.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(nn_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    # Print mean, standard deviation and parameters of each combination of parameters
    means = nn_cv.cv_results_['mean_test_score']
    stds = nn_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, nn_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # Print classification report using validation data
    y_true, y_pred = y_val, nn_cv.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    # Set Nearest Neighbour Classifier model with best parameters
    nearest_neighbour_classifier = KNeighborsClassifier(n_neighbors=nn_cv.best_params_['n_neighbors'],
                                                        leaf_size=nn_cv.best_params_['leaf_size'])

    # Train Nearest Neighbour Classifier model with training dataset
    nearest_neighbour_classifier.fit(X_training, y_training)

    # Return Nearest Neighbour Classifier model
    return nearest_neighbour_classifier


def nearest_neighbours_test(nn_model, X_test, y_test):
    """
    Predicts data based on a X_test dataset using Nearest Neighbour model. Compares results with y_test data and returns
    an MAE and Accuracy Score, along with a confusion matrix and heatmap for visualisation.
    :param nn_model: A trained Nearest Neighbour model. Trained in build_NearrestNeighbours_classifier function
    :param X_test: X value test set
    :param y_test: y value test set

    """
    # Predict y_values of test set
    val_predictions_nn = nn_model.predict(X_test)

    print("MAE for Nearest Neighbour Classifier: {}".format(mean_absolute_error(y_test, val_predictions_nn)))
    print("Accuracy Score for Nearest Neighbour  Classifier: {}".format(accuracy_score(y_test, val_predictions_nn)))

    # Create confusion matrix with right and wrong estimations
    cm = confusion_matrix(y_test, val_predictions_nn)

    # Create matplotlib figure
    nn_plot = plt.axes()

    # Use seaborn library to create heatmap
    sns.heatmap(cm, annot=True, fmt='d', ax=nn_plot)

    # Set figure title
    nn_plot.set_title("Nearest Neighbour Heatmap")

    # Show figure
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_SupportVectorMachine_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    :param X_training: X_training[i,:] is the ith example
    :param y_training: y_training[i] is the class label of X_training[i,:]

    :return the classifier built in this function
    """
    # Split Training Dataset into, Train and Validate Datasets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    # Define parameters to be tuned by GridSearchCV
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

    print("# Tuning hyper-parameters for precision")
    print()

    # Find best parameters to use based on tuned_parameters. Score on precision
    svm_cv = GridSearchCV(
        SVC(), tuned_parameters, scoring='precision', n_jobs=-1
    )

    # Fit model to train data
    svm_cv.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(svm_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    # Print mean, standard deviation and parameters of each combination of parameters
    means = svm_cv.cv_results_['mean_test_score']
    stds = svm_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svm_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # Print classification report using validation data
    y_true, y_pred = y_val, svm_cv.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    # Set Support Vector Machine Classifier model with best parameters
    svm_classifier = SVC(kernel=svm_cv.best_params_['kernel'], C=svm_cv.best_params_['C'], random_state=1)

    # Train Support Vector Machine Classifier model with training dataset
    svm_classifier.fit(X_training, y_training)

    # Return Support Vector Machine Classifier model
    return svm_classifier


def svm_test(svm_model, X_test, y_test):
    """
    Predicts data based on a X_test dataset using SVM model. Compares results with y_test data and returns
    an MAE and Accuracy Score, along with a confusion matrix and heatmap for visualisation.

    :param svm_model: A trained SVM model. Trained in build_SupportVectorMachine_classifier function
    :param X_test: X value test set
    :param y_test: y value test set

    """
    # Predict y_values of test set
    val_predictions_svm = svm_model.predict(X_test)

    print("MAE for SVM Classifier: {}".format(mean_absolute_error(y_test, val_predictions_svm)))
    print("Accuracy Score for SVM Classifier: {}".format(accuracy_score(y_test, val_predictions_svm)))

    # Create confusion matrix with right and wrong estimations
    cm = confusion_matrix(y_test, val_predictions_svm)

    # Create matplotlib figure
    svm_plot = plt.axes()

    # Use seaborn library to create heatmap
    sns.heatmap(cm, annot=True, fmt='d', ax=svm_plot)

    # Set figure title
    svm_plot.set_title("SVM Heatmap")

    # Show figure
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NN:
    """
    Neural Network Class used to initialise and create a neural network using Keras and Tensorflow.
    Implemented to prevent thread errors caused by running two neural networks simultaneously with GridSearchCV
    """
    def __init__(self):
        """
        Initialise Neural Network Class
        """
        pass

    def create_model(self, neurons_1=16, neurons_2=8):
        """
        Creates the neural network based on a set of parameters. Used for GridSearchCV and final fitting of model
        with best neurons numbers.

        :param neurons_1: Number of neurons used for hidden layer 1
        :param neurons_2: Number of neurons used for hidden layer 2
        :return: Compiled neural network model
        """

        # Neural Network initialisation
        nn_model = Sequential()

        # Add hidden dense layer with prescribed number of neurons_1
        nn_model.add(Dense(neurons_1, kernel_initializer='uniform', activation='relu', input_dim=30))

        # Add hidden dense layer with prescribed number of neurons_2
        nn_model.add(Dense(neurons_2, kernel_initializer='uniform', activation='relu'))

        # Add final layer with set parameters
        nn_model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        # Compile neural network with set parameters
        nn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return nn_model


def build_NeuralNetwork_classifier(X_training, y_training):
    """
    Build a Neural Network classifier (with two dense hidden layers)
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    :param X_training: X_training[i,:] is the ith example
    :param y_training: y_training[i] is the class label of X_training[i,:]

    :return the Neural Network classifier built
    """
    # Split Training Dataset into, Train and Validate Datasets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    # Initialise Neural Network class
    nn = NN()
    # Build KerasClassifier with neural network created. Used for GridSearchCV
    keras_model = KerasClassifier(build_fn=nn.create_model, batch_size=4, epochs=50, verbose=0)

    # Define parameters to be tuned by GridSearchCV
    neurons_1 = [1, 5, 10, 15, 20, 25, 30]
    neurons_2 = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons_1=neurons_1, neurons_2=neurons_2)

    # Find best parameters to use based on tuned_parameters. Score on precision
    neural_network_cv = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='Precision')

    # Fit model to train data
    neural_network_cv.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(neural_network_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    # Print mean, standard deviation and parameters of each combination of parameters
    means = neural_network_cv.cv_results_['mean_test_score']
    stds = neural_network_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, neural_network_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # Print classification report using validation data
    y_true, y_pred = y_val, neural_network_cv.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    # Initialise new Neural Network and create with best number of neurons.
    nn = NN()
    nn_model = nn.create_model(neural_network_cv.best_params_['neurons_1'], neural_network_cv.best_params_['neurons_2'])

    # Train Neural Network Classifier model with training dataset
    nn_model.fit(X_training, y_training, epochs=50, batch_size=1)

    # Return Neural Network Classifier model
    return nn_model


def neural_Network_test(neural_network_classifier, X_test, y_test):
    """
    Predicts data based on a X_test dataset using Neural Network model. Compares results with y_test data and returns
    an MAE and Accuracy Score, along with a confusion matrix and heatmap for visualisation.

    :param neural_network_classifier: Neural Network Model
    :param X_test: X_test
    :param y_test: y_test
    """
    # Predict y_values of test set
    neural_network_classifier_pred = neural_network_classifier.predict(X_test)
    print("MAE for Neural Network Classifier: {}".format(mean_absolute_error(y_test, neural_network_classifier_pred)))
    print("Accuracy Score for Neural Network Classifier: {}".format(
        neural_network_classifier.evaluate(X_test, y_test)[1]))

    # Convert floats to integers and convert to numpy array
    pred = []
    for nums in neural_network_classifier_pred:
        pred.append(round(nums))
    pred_np = np.array(pred)

    # Create confusion matrix with right and wrong estimations
    cm_2 = confusion_matrix(y_test, pred_np)

    # Create matplotlib figure
    neuralnet_plot = plt.axes()

    # Use seaborn library to create heatmap
    sns.heatmap(cm_2, annot=True, fmt='d', ax=neuralnet_plot)

    # Set figure title
    neuralnet_plot.set_title("Neural Network Heatmap")

    # Show figure
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    path = "medical_records.data"
    X, y = prepare_dataset(path)
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # decision_tree_classifier = build_DecisionTree_classifier(X_training, y_training)
    # decision_tree_test(decision_tree_classifier, X_test, y_test)
    #
    # nearest_neighbour_classifier = build_NearrestNeighbours_classifier(X_training, y_training)
    # nearest_neighbours_test(nearest_neighbour_classifier, X_test, y_test)

    # svm_classifier = build_SupportVectorMachine_classifier(X_training, y_training)
    # svm_test(svm_classifier, X_test, y_test)
    #
    neural_network_classifier = build_NeuralNetwork_classifier(X_training, y_training)
    neural_Network_test(neural_network_classifier, X_test, y_test)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
