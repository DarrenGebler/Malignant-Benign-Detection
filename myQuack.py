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
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    tuned_parameters = [{'criterion': ['gini'], 'max_depth': [1, 5, 25, 50, 100, 250, 500],
                         'max_leaf_nodes': [None, 5, 25, 50, 100, 250, 500], 'random_state': [1]}]

    print("# Tuning hyper-parameters for precision")
    print()

    clf = GridSearchCV(
        DecisionTreeClassifier(), tuned_parameters, scoring='precision'
    )
    clf.fit(X_train, y_train)

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
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val, clf.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    decision_tree_classifier = DecisionTreeClassifier(criterion=clf.best_params_['criterion'],
                                                      max_depth=clf.best_params_['max_depth'],
                                                      max_leaf_nodes=clf.best_params_['max_leaf_nodes'], random_state=1)

    decision_tree_classifier.fit(X_training, y_training)

    return decision_tree_classifier


def decision_tree_test(dt_model, X_test, y_test):
    val_predictions_dt = dt_model.predict(X_test)
    print("MAE for Decision Tree Classifier: {}".format(mean_absolute_error(y_test, val_predictions_dt)))
    print("Accuracy Score for Decision Tree Classifier: {}".format(accuracy_score(y_test, val_predictions_dt)))
    cm_2 = confusion_matrix(y_test, val_predictions_dt)
    dt_plot = plt.axes()
    sns.heatmap(cm_2, annot=True, fmt='d', ax=dt_plot)
    dt_plot.set_title("Decision Tree Heatmap")
    plt.show()


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
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    tuned_parameters = [
        {'n_neighbors': [1, 3, 5, 10, 15, 20, 30, 50, 100], 'leaf_size': [1, 3, 5, 10, 15, 20, 30, 50, 100]}]

    print("# Tuning hyper-parameters for %s" % 'precision')
    print()

    clf = GridSearchCV(
        KNeighborsClassifier(), tuned_parameters, scoring='precision'
    )
    clf.fit(X_train, y_train)

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
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val, clf.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    nearest_neighbour_classifier = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'],
                                                        leaf_size=clf.best_params_['leaf_size'])

    nearest_neighbour_classifier.fit(X_training, y_training)

    return nearest_neighbour_classifier


def nearest_neighbours_test(nn_model, X_test, y_test):
    val_predictions_nn = nn_model.predict(X_test)
    print("MAE for Nearest Neighbour Classifier: {}".format(mean_absolute_error(y_test, val_predictions_nn)))
    print("Accuracy Score for Nearest Neighbour  Classifier: {}".format(accuracy_score(y_test, val_predictions_nn)))
    cm_2 = confusion_matrix(y_test, val_predictions_nn)
    nn_plot = plt.axes()
    sns.heatmap(cm_2, annot=True, fmt='d', ax=nn_plot)
    nn_plot.set_title("Nearest Neighbour Heatmap")
    plt.show()


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
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

    print("# Tuning hyper-parameters for precision")
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='precision', n_jobs=-1
    )
    clf.fit(X_train, y_train)

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
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val, clf.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    svm_classifier = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], random_state=1)

    svm_classifier.fit(X_training, y_training)

    return svm_classifier


def svm_test(svm_model, X_test, y_test):
    val_predictions_svm = svm_model.predict(X_test)
    print("MAE for SVM Classifier: {}".format(mean_absolute_error(y_test, val_predictions_svm)))
    print("Accuracy Score for SVM Classifier: {}".format(accuracy_score(y_test, val_predictions_svm)))
    cm_2 = confusion_matrix(y_test, val_predictions_svm)
    svm_plot = plt.axes()
    sns.heatmap(cm_2, annot=True, fmt='d', ax=svm_plot)
    svm_plot.set_title("SVM Heatmap")
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class NN:
    def __init__(self):
        pass

    def create_model(self, neurons_1=16, neurons_2=8):
        model = Sequential()
        model.add(Dense(neurons_1, kernel_initializer='uniform', activation='relu', input_dim=30))
        model.add(Dense(neurons_2, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model



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

    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=2)

    nn = NN()
    keras_model = KerasClassifier(build_fn=nn.create_model, batch_size=4, epochs=50, verbose=0)
    neurons_1 = [1, 5, 10, 15, 20, 25, 30]
    neurons_2 = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons_1=neurons_1, neurons_2=neurons_2)
    neural_network = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, n_jobs=-1)
    neural_network.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(neural_network.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = neural_network.cv_results_['mean_test_score']
    stds = neural_network.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, neural_network.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val, neural_network.predict(X_val)
    print(classification_report(y_true, y_pred))
    print()

    nn = NN()
    nn_model = nn.create_model(neural_network.best_params_['neurons_1'], neural_network.best_params_['neurons_2'])
    nn_model.fit(X_training, y_training, epochs=50, batch_size=1)

    return nn_model


def neural_Network_test(neural_network_classifier, X_test, y_test):
    neural_network_classifier_pred = neural_network_classifier.predict(X_test)
    print("MAE for Neural Network Classifier: {}".format(mean_absolute_error(y_test, neural_network_classifier_pred)))
    print("Accuracy Score for Neural Network Classifier: {}".format(
        neural_network_classifier.evaluate(X_test, y_test)))
    pred = []
    for nums in neural_network_classifier_pred:
        if nums > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    pred_np = np.array(pred)
    cm_2 = confusion_matrix(y_test, pred_np)
    neuralnet_plot = plt.axes()
    sns.heatmap(cm_2, annot=True, fmt='d', ax=neuralnet_plot)
    neuralnet_plot.set_title("Neural Network Heatmap")
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    path = "medical_records.data"
    X, y = prepare_dataset(path)
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    decision_tree_classifier = build_DecisionTree_classifier(X_training, y_training)
    decision_tree_test(decision_tree_classifier, X_test, y_test)

    nearest_neighbour_classifier = build_NearrestNeighbours_classifier(X_training, y_training)
    nearest_neighbours_test(nearest_neighbour_classifier, X_test, y_test)

    svm_classifier = build_SupportVectorMachine_classifier(X_training, y_training)
    svm_test(svm_classifier, X_test, y_test)

    neural_network_classifier = build_NeuralNetwork_classifier(X_training, y_training)
    neural_Network_test(neural_network_classifier, X_test, y_test)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
