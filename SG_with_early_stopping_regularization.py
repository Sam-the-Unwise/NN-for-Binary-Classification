###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Kruse
# DESCRIPTION: program that will implement a stochastic gradient descent algo
#       for a neural network with one hidden layer
# VERSION: 1.3.0v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss
from sklearn.metrics import log_loss
from sklearn import neighbors, datasets
from matplotlib import pyplot as plt
import random
from KNN_KFold import KFoldCV, NearestNeighborsCV_Wrapper
from gradient_descent import gradient_descent
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# global variables
MAX_EPOCHS = 750
STEP_SIZE = .01
N_HIDDEN_UNITS = 10

# Function: NNOneSplit
# INPUT ARGS:
#       X_mat : feature matrix (n_observations x n_features)
#       y_vec : label vector (n_observations x 1)
#       max_epochs : scalar int > 1
#       step_size
#       n_hidden_units : number of hidden units
#       is_train : logical vector of size n_observations
#           - TRUE if the observation is the train set
#           - FALSE for the validation set
# Return: list/dictionary containing at least three elements named:
#           - v_mat : best weight matrix (n_features x n_hidden_units) used to
#                          predict hidden units given inputs
#           - w_vec : best weight vector (n_hidden_units) used to predict output
#                          given hidden units
#           - loss_values : a matrix/data_table/etc which stores the logistic
#                          loss with respect to the train/validation set for each
#                          iteration
def NNOneSplit( X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_train ):
    n_features = X_mat.shape[1]

    #   for is_train, randomly assign 60% train and 40% validation
    #X_train, X_validation, y_train, y_validation = split_matrix(X_mat, y_vec, .6)
    X_train = np.delete( X_mat, np.argwhere(is_train != True), 0)
    y_train = np.delete(y_vec, np.argwhere(is_train != True), 0)
    X_validation = np.delete(X_mat, np.argwhere(is_train != False), 0)
    y_validation = np.delete(y_vec, np.argwhere(is_train != False), 0)

    np.random.seed(0)
    v_mat = np.random.randn(n_features, n_hidden_units)
    w_vec = np.random.randn(n_hidden_units)

    loss_train = []
    loss_val = []
    # during each iteration compute the gradients of v_mat and w_vec
    #       by taking a step (scaled by step_size) in the neg. gradient direction-
    for epoch in range(max_epochs):
        if (epoch % 10 == 0): print("Iteration ", epoch, "of ", max_epochs)
        for index in range((X_train.shape[0]) - 1):
            if y_train[index] == 0:
                y_tild = -1
            else:
                y_tild = 1

            h_v = sigmoid(np.transpose(v_mat) @ X_train[index])

            first_term = (1 / (1 + np.exp(-y_tild * (h_v))))
            second_term = (np.exp(-y_tild * (h_v)))
            third_term = (-y_tild * (np.transpose(v_mat * (h_v * (1 - h_v))) @ X_train[index]))
            gradient = (first_term * second_term * third_term)
            v_mat = v_mat - step_size * gradient

            # print(v_mat)
            y_hat = sigmoid(w_vec @ h_v)  # TODO: this shouldn't be this way :0
            # print(y_hat)

            first_term = (1 / (1 + np.exp(-y_tild * (y_hat))))
            second_term = (np.exp(-y_tild * (y_hat)))
            third_term = (-y_tild * (np.transpose(w_vec * (y_hat * (1 - y_hat))) @ h_v))
            gradient = (first_term * second_term * third_term)
            w_vec = w_vec - step_size * gradient

        # at each iteration compute the log. loss on the train/validation sets
        #       store in loss_values
        y_train_pred = np.around(sigmoid(sigmoid(X_train @ v_mat) @ w_vec))
        # print(y_train_pred)
        # print("hi", np.mean( y_train != y_train_pred ))
        # loss_train.append( np.mean( y_train != y_train_pred ) )
        loss_train.append(log_loss(y_train, y_train_pred))

        y_val_pred = np.around(sigmoid(sigmoid(X_validation @ v_mat) @ w_vec))
        # print(y_val_pred)
        # print("bye", np.mean( y_validation != y_val_pred))
        # loss_val.append( np.mean( y_validation != y_val_pred) )
        if (y_validation.shape[0] > 0):
            loss_val.append(log_loss(y_validation, y_val_pred))
        else:
            loss_val.append([])

    loss_values = [[],[]]
    loss_values[0].append(loss_train)
    loss_values[1].append(loss_val)
    #print(loss_values)

    return v_mat, w_vec, loss_values

# Function: split matrix
# INPUT ARGS:
#   X_mat : matrix to be split
#   y_vec : corresponding vector to X_mat
# Return: train, validation, test
def split_matrix(X_mat, y_vec, size):
    # split data 80% train by 20% validation
    X_train, X_validation = np.split( X_mat, [int(size * len(X_mat))])
    y_train, y_validation = np.split( y_vec, [int(size * len(y_vec))])

    return (X_train, X_validation, y_train, y_validation)


# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full

# Function: sigmoid
# INPUT ARGS:
#   x : value to be sigmoidified
# Return: sigmoidified x
def sigmoid(x) :
    x = 1 / (1 + np.exp(-x))
    return x

def BaselineFunction( X_mat, y_vec, X_new ) :
    X_new = np.array(X_new)
    return np.zeros(X_new.shape[0])

def NNOneSplit_wrap( X_mat, y_vec, X_new ) :
    #is_subtrain = np.random.choice( [True, False], y_vec.shape[0], p=[.6, .4] )
    #v_mat, w_vec, loss_values = NNOneSplit(X_mat, y_vec, MAX_EPOCHS, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain)
    best_epochs = 559
    is_subtrain = np.full( y_vec.shape, True)
    v_mat, w_vec, loss_values = NNOneSplit( X_mat, y_vec, best_epochs, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain )

    return np.around(sigmoid(sigmoid(X_new @ v_mat) @ w_vec))

def Gradient_Wrapper( X_mat, y_vec, X_new):
    weightVec = gradient_descent( X_mat, y_vec, 0.1, 600 )
    return np.dot(X_new, weightVec)

# Function: main
def main():
    print("starting")
    # use spam data set
    data_matrix_full = convert_data_to_matrix("spam.data")
    np.random.seed( 0 )
    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    y_vec = data_matrix_full[:,57]

    # First scale the input matrix (each column should have mean 0 and variance 1).
    # You can do this by subtracting away the mean and then dividing by the standard deviation of each column
    # (or just use a standard function like scale in R).
    X_sc = scale(X_Mat)
    '''
    # (5 points) Next create a variable is.train (logical vector with size equal to the number of observations
    # in the whole data set). Each element is TRUE if the corresponding observation (row of input matrix)
    # is in the train set, and FALSE otherwise.
    # There should be 80% train, 20% test observations (out of all observations in the whole data set).
    is_train = np.random.choice( [True, False], X_sc.shape[0], p=[.8, .2] )

    # (5 points) Next create a variable is.subtrain (logical vector with size equal to the
    # number of observations in the train set).
    # Each element is TRUE if the corresponding observation is is the subtrain set, and FALSE otherwise.
    # There should be 60% subtrain, 40% validation observations (out of 100% train observations).
    subtrain_size = np.sum( is_train==True )
    is_subtrain = np.random.choice( [True, False], subtrain_size, p=[.6, .4] )

    # (5 points) Use NNOneSplit with the train set as X.mat/y.vec,
    # with is.subtrain as specified above, and a large number for max.epochs.
    X_train = np.delete( X_sc, np.argwhere(is_train != True), 0)
    y_train = np.delete(y_vec, np.argwhere(is_train != True), 0)

    v_mat, w_vec, loss_values = NNOneSplit( X_train, y_train, MAX_EPOCHS, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain )

    # (5 points) Plot the subtrain/validation loss as a function of the number of epochs,
    # and draw a point to emphasize the minimum of the validation loss curve.
    print(loss_values[0])
    plt.plot(loss_values[0][0], "-g", label="Train")
    min_index = np.argmin(loss_values[0][0])
    plt.plot(min_index, loss_values[0][0][min_index], "go")

    plt.plot(loss_values[1][0], "-r", label="Validation")
    min_index = np.argmin(loss_values[1][0])
    plt.plot(min_index, loss_values[1][0][min_index], "ro")
    plt.xlabel('epoch')
    plt.ylabel('logistic loss')

    plt.xlim(0)
    plt.legend()
    plt.show()

    # (5 points) Define a variable called best_epochs which is the number of epochs which minimizes the validation loss.
    # Use NNOneSplit with the train set as X.mat/y.vec,
    # but this time use is.subtrain=TRUE for all observations, and use max.epochs=best_epochs.
    best_epochs = min_index
    print(best_epochs)
    is_subtrain = np.full( y_vec.shape, True)
    v_mat, w_vec, loss_values = NNOneSplit( X_sc, y_vec, best_epochs, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain )

    plt.plot(loss_values[0][0], "-g", label="Train")
    min_index = np.argmin(loss_values[0][0])
    plt.plot(min_index, loss_values[0][0][min_index], "go")


    plt.xlabel('epoch')
    plt.ylabel('error rate') 

    
    plt.xlim(0)
    plt.legend()
    plt.show()
    
    # (5 points) Finally use the learned V.mat/w.vec to make predictions on the test set.
    # What is the prediction accuracy? (percent correctly predicted labels in the test set)
    # What is the prediction accuracy of the baseline model which predicts the most frequent class in the train labels?
    X_test = np.delete( X_sc, np.argwhere(is_train==False), 0)
    y_test = np.delete(y_vec, np.argwhere(is_train == False), 0)

    test_pred = np.around(sigmoid(sigmoid( X_test @ v_mat ) @ w_vec))
    print("Prediction accuracy (correctly labeled) :", np.mean( test_pred == y_test ))
    baseline = np.zeros(y_test.shape)
    print("Baseline prediction accuracy :", np.mean( baseline == y_test ))'''

    ########### EXTRA CREDIT #########
    test_fold_vec = np.random.randint(1, 4 + 1, X_sc.shape[0])

    print("Calculating baseline")
    result_base = KFoldCV(X_sc, y_vec, BaselineFunction, test_fold_vec)
    base_labels = ['Base', 'Base', 'Base', 'Base']
    plt.scatter(result_base[0], base_labels)

    print("Calculating gradient")
    result_grad = KFoldCV(X_sc, y_vec, Gradient_Wrapper, test_fold_vec)
    grad_labels = ['Gradient', 'Gradient', 'Gradient', 'Gradient']
    plt.scatter(result_grad[0], grad_labels)

    print("Calculating NN")
    result_NN = KFoldCV(X_sc, y_vec, NNOneSplit_wrap, test_fold_vec)
    NN_labels = ['Neural', 'Neural', 'Neural', 'Neural']
    plt.scatter(result_NN[0], NN_labels)

    print("Calculating Nearest Neighbor")
    result_nearest = KFoldCV(X_sc, y_vec, NearestNeighborsCV_Wrapper, test_fold_vec)
    nearest_labels = ['Nearest', 'Nearest', 'Nearest', 'Nearest']
    plt.scatter(result_nearest[0], nearest_labels)

    plt.title("Percent of Error Comparison")
    plt.xlim(left=0, right=1)
    plt.show()

    base_label = "Baseline"
    NN_label = "Single Layer Neural Network"
    nearest_label = "Nearest Neighbor"
    grad_label = "Gradient Descent"


    for index in range(4):
        fpr, tpr, thr = roc_curve(result_base[1][index], result_base[2][index])
        plt.plot(fpr, tpr, "r-", linewidth=".4", label=base_label)

        fpr, tpr, thr = roc_curve(result_NN[1][index], result_NN[2][index])
        plt.plot(fpr, tpr, "b-", linewidth=".6", label=NN_label)
        plt.plot(fpr[int(len(fpr) / 2)], tpr[int(len(tpr) / 2)], "b.")

        fpr, tpr, thr = roc_curve(result_nearest[1][index], result_nearest[2][index])
        plt.plot(fpr, tpr, "g-", linewidth=".4", label=nearest_label)
        plt.plot(fpr[int(len(fpr) / 2)], tpr[int(len(tpr) / 2)], "g.")

        fpr, tpr, thr = roc_curve(result_grad[1][index], result_grad[2][index])
        plt.plot(fpr, tpr, "c-", linewidth=".4", label=grad_label)
        plt.plot(fpr[int(len(fpr) / 2)], tpr[int(len(tpr) / 2)], "c.")

        base_label = "_nolegend_"
        NN_label = "_nolegend_"
        nearest_label = "_nolegend_"
        grad_label = "_nolegend_"

    plt.title("ROC curves, 4-fold CV")
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.legend()
    plt.show()

    for index in range(4):
        score_base = roc_auc_score(result_base[1][index], result_base[2][index])
        plt.scatter(score_base, "Baseline", c="r")

        score_over = roc_auc_score(result_NN[1][index], result_NN[2][index])
        plt.scatter(score_over, "Neural Network", c="g")

        score_nearest = roc_auc_score(result_nearest[1][index], result_nearest[2][index])
        plt.scatter(score_nearest, "Nearest", c="b")

        score_grad = roc_auc_score(result_grad[1][index], result_grad[2][index])
        plt.scatter(score_grad, "Gradient", c="c")

    plt.title("ROC Area Under Curve Comparison")
    plt.xlim(right=1)
    plt.show()

    return 0

main()
