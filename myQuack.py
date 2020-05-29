'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import RFECV, SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns


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
    column_headers = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
                      'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry',
                      'Mean Fractal', 'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness',
                      'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal',
                      'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
                      'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal']

    data = pd.read_csv(dataset_path, names=column_headers)

    X = data.iloc[:, 2:32]
    y = data.iloc[:, 1]

    diagnosis_encoder = LabelEncoder()
    y = diagnosis_encoder.fit_transform(y)
    print(data.columns)

    return X, y


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

    tuned_parameters = [{'criterion': ['gini'], 'max_depth': [1, 5, 25, 50, 100, 250, 500],
                         'max_leaf_nodes': [None, 5, 25, 50, 100, 250, 500], 'random_state': [1]}]

    print("# Tuning hyper-parameters for precision")
    print()

    clf = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters, scoring='%s_macro' % 'precision'
    )
    clf.fit(X_training, y_training)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


    decision_tree_classifier = DecisionTreeClassifier(criterion=clf.best_params_['criterion'], max_depth=clf.best_params_['max_depth'],
                                                      max_leaf_nodes=clf.best_params_['max_leaf_nodes'], random_state=1)

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

    tuned_parameters = [
        {'n_neighbors': [1, 3, 5, 10, 15, 20, 30, 50, 100], 'leaf_size': [1, 3, 5, 10, 15, 20, 30, 50, 100]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            KNeighborsClassifier(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_training, y_training)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    nearest_neighbour_classifier = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'], leaf_size=clf.best_params_['leaf_size'])

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

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_training, y_training)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    svm_classifier = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], random_state=1)

    svm_classifier.fit(X_training, y_training)

    return svm_classifier



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

    model = Sequential()
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_training, y_training, epochs=100, batch_size=1)

    return model

def neural_Network_testing(neural_network_classifier, X_val, y_val):
    neural_network_classifier_pred = neural_network_classifier.predict(X_val)
    print("MAE for Neural Network Classifier: {}".format(mean_absolute_error(y_val, neural_network_classifier_pred)))
    print("Accuracy Score for Neural Network Classifier: {}".format(accuracy_score(y_val, neural_network_classifier_pred)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    path = "medical_records.data"
    X, y = prepare_dataset(path)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_training, X_val, y_training, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    decision_tree_classifier = build_DecisionTree_classifier(X, y)
    # nearest_neighbour_classifier = build_NearrestNeighbours_classifier(X_training, y_training)
    # svm_classifier = build_SupportVectorMachine_classifier(X_training, y_training)

    val_predictions_dt = decision_tree_classifier.predict(X_val)
    # val_predictions_nn = nearest_neighbour_classifier.score(X_val, y_val)
    # val_predictions_svm = svm_classifier.predict(X_val)

    print("MAE for Decision Tree Classifier: {}".format(mean_absolute_error(y_val, val_predictions_dt)))
    print("Accuracy Score for Decision Tree Classifier: {}".format(accuracy_score(y_val, val_predictions_dt)))
    cm_2 = confusion_matrix(y_test, decision_tree_classifier.predict(X_test))
    sns.heatmap(cm_2, annot=True, fmt='d')
    plt.show()
    # print("MAE for Nearest Neighbour Classifier: {}".format(val_predictions_nn))
    # print("MAE for SVM Classifier: {}".format(mean_absolute_error(y_val, val_predictions_svm)))

    # neural_network_classifier = build_NeuralNetwork_classifier(X_training, y_training)
    # #
    # neural_network_classifier_pred = neural_network_classifier.predict(X_val)
    # #
    # print("MAE for Neural Network Classifier: {}".format(mean_absolute_error(y_val, neural_network_classifier_pred)))

    # score = decision_tree_classifier.evaluate(X_val, y_val, batch_size=16)
    # print("Score = {}".format(score))

    # print(decision_tree_classifier.score(X_val, y_val))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# if __name__ == "__main__":
#     pass
#     # Write a main part that calls the different
#     # functions to perform the required tasks and repeat your experiments.
#     # Call your functions here
#
#     ##         "INSERT YOUR CODE HERE"
#     raise NotImplementedError()
