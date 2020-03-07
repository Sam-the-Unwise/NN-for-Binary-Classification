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

# global variables
MAX_EPOCHS = 650
STEP_SIZE = .01
N_HIDDEN_UNITS = 10

NUM_FOLDS = 4



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
#       fold_vector : a vector of integer fold ID numbers (from 1 to K).
# Return: list/dictionary containing at least three elements named:
#           - v_mat : best weight matrix (n_features x n_hidden_units) used to
#                          predict hidden units given inputs
#           - w_vec : best weight vector (n_hidden_units) used to predict output
#                          given hidden units
#           - loss_values : a matrix/data_table/etc which stores the logistic
#                          loss with respect to the train/validation set for each
#                          iteration
def NNOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_train, fold_vector):
    n_features = X_mat.shape[1]
    #print(n_features)

    # initialize v_mat and w_vec to some random number close to zero
    np.random.seed( 0 )
    v_mat = np.random.randn( n_features, n_hidden_units )
    w_vec = np.random.randn( n_hidden_units )
    #print(v_mat)
    #print(w_vec)

    col = X_mat.shape[1]
    row = X_mat.shape[0]

    array_of_zeros = [0] * (col)

    #print(v_mat.shape[1])

    loss_train = {}
    loss_val = {}
    
    for foldIDk in range(1, NUM_FOLDS + 1):
        # create arrays that will hold the loss_train and loss_val outputs for this foldIDk
        sub_loss_val = []
        sub_loss_train = []

        # define X_train, y_train using all the other observations
        X_train = np.array(array_of_zeros).reshape(1, col)
        y_train = np.zeros(1)

        # define X_validation, y_validation to contain the corr. folds of foldIDk to num_folds
        X_validation = np.array(array_of_zeros).reshape(1, col)
        y_validation = np.zeros(1)

        one_items_in_X_validation = 0


        new_index = 0
        train_index = 0

        # define X_validation, y_validation based on the observations for which the corr.
        #       elements of fold_vec are equal to the current fold ID k
        index = 0
        for num in fold_vector:

            if num == foldIDk:
                # # account for possibility that our row amount doesn't divide evenly into 80% and 20%
                # # ex: 4601 yields 80%: 3680 and 20%:920, meaning we didn't account for the last 1

                y_validation = np.append(y_validation, y_vec[index])
                X_validation = np.vstack((X_validation, X_mat[index,:]))

                new_index += 1

            else:

                y_train = np.append(y_train, y_vec[index])
                X_train = np.vstack((X_train, X_mat[index,:]))

                train_index += 1

            index += 1

        # delete zero elements used to initialize the arrays
        X_validation = np.delete(X_validation, 1, 0)
        X_train = np.delete(X_train, 1, 0)
        y_train = np.delete(y_train, 1, 0)
        y_validation = np.delete(y_validation, 1, 0)

        #is_train = np.array([])

        X_train_i = X_train[0]

        index = 0

        # during each iteration compute the gradients of v_mat and w_vec
        #       by taking a step (scaled by step_size) in the neg. gradient direction-
        for epoch in range(max_epochs):
            
            if( epoch%10 == 0 ) : print("Iteration ", epoch, "of ", max_epochs)
            
            for index in range( (X_train.shape[0]) -1 ) :
                if y_train[ index ] == 0 : y_tild = -1
                else : y_tild = 1

                h_v = sigmoid( np.transpose( v_mat ) @ X_train[ index ] )

                first_term = (1 / (1 + np.exp(-y_tild * (h_v))))
                second_term = (np.exp(-y_tild * (h_v)))
                third_term = (-y_tild * (np.transpose(v_mat * (h_v * (1 - h_v))) @ X_train[index]))
                gradient = (first_term * second_term * third_term)
                v_mat = v_mat - step_size * gradient

                #print(v_mat)
                y_hat = sigmoid( w_vec @ h_v ) # TODO: this shouldn't be this way :0
                #print(y_hat)

                first_term = (1 / (1 + np.exp(-y_tild * (y_hat))))
                second_term = (np.exp(-y_tild * (y_hat)))
                third_term = (-y_tild * (np.transpose(w_vec * (y_hat * (1 - y_hat))) @ h_v))
                gradient = (first_term * second_term * third_term)
                w_vec = w_vec - step_size * gradient

            # at each iteration compute the log. loss on the train/validation sets
            #       store in loss_values
            y_train_pred = np.around(sigmoid(sigmoid(X_train @ v_mat) @ w_vec ))
            sub_loss_train.append( log_loss( y_train, y_train_pred ))

            y_val_pred = np.around(sigmoid(sigmoid(X_validation @ v_mat) @ w_vec))
            sub_loss_val.append( log_loss( y_validation, y_val_pred ))
        
        loss_train[foldIDk] = sub_loss_train
        loss_val[foldIDk] = sub_loss_val

    loss_values = [[],[]]
    loss_values[0].append(loss_train)
    loss_values[1].append(loss_val)

    # NOTE: loss_values looks like [[{1: [], 2: [],..., k: []}], [{1: [], 2: [],..., k: []}]]

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

    multiplier_of_num_folds = int(X_Mat.shape[0]/NUM_FOLDS)

    extra_folds_to_add = 0

    fold_vec = np.array(list(np.arange(1, NUM_FOLDS + 1)) * multiplier_of_num_folds)

    # (5 points) Use NNOneSplit with the train set as X.mat/y.vec,
    # with is.subtrain as specified above, and a large number for max.epochs.
    # NOTE: loss_values looks like [[{1: [], 2: [],..., k: []}], [{1: [], 2: [],..., k: []}]]
    v_mat, w_vec, loss_values = NNOneSplit(X_sc, y_vec, MAX_EPOCHS, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain, fold_vec)

    # (5 points) Plot the subtrain/validation loss as a function of the number of epochs,
    # and draw a point to emphasize the minimum of the validation loss curve.

    dict_of_train_loss = loss_values[0][0]
    dict_of_val_loss = loss_values[1][0]
    
    for foldID in range(1, NUM_FOLDS):
        print(dict_of_train_loss[foldID])
        print("")
        print(dict_of_val_loss[foldID])
        print("\n\n")


    # print(loss_values[0])
    # plt.plot(loss_values[0][0], "-g", label="Train")
    # min_index = np.argmin(loss_values[0][0])
    # plt.plot(min_index, loss_values[0][0][min_index], "go")

    # plt.plot(loss_values[1][0], "-r", label="Validation")
    # min_index = np.argmin(loss_values[1][0])
    # plt.plot(min_index, loss_values[1][0][min_index], "ro")

    # plt.xlim(0)
    # plt.legend()
    # plt.show()

    # # (5 points) Define a variable called best_epochs which is the number of epochs which minimizes the validation loss.
    # # Use NNOneSplit with the train set as X.mat/y.vec,
    # # but this time use is.subtrain=TRUE for all observations, and use max.epochs=best_epochs.
    # best_epochs = min_index
    # print(best_epochs)
    # is_subtrain = np.full( is_subtrain.shape, True)
    # v_mat, w_vec, loss_values = NNOneSplit( X_sc, y_vec, best_epochs, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain )

    # plt.plot(loss_values[0][0], "-g", label="Train")
    # min_index = np.argmin(loss_values[0][0])
    # plt.plot(min_index, loss_values[0][0][min_index], "go")

    # plt.plot(loss_values[1][0], "-r", label="Validation")
    # min_index = np.argmin(loss_values[1][0])
    # plt.plot(min_index, loss_values[1][0][min_index], "ro")
    # plt.xlim(0)
    # plt.legend()
    # plt.show()

    # # (5 points) Finally use the learned V.mat/w.vec to make predictions on the test set.
    # # What is the prediction accuracy? (percent correctly predicted labels in the test set)
    # # What is the prediction accuracy of the baseline model which predicts the most frequent class in the train labels?
    # X_test = np.delete( X_sc, np.argwhere(is_train==False), 0)
    # y_test = np.delete(y_vec, np.argwhere(is_train == False), 0)

    # test_pred = np.around(sigmoid(sigmoid( X_test @ v_mat ) @ w_vec))
    # print("Prediction accuracy (correctly labeled) :", np.mean( test_pred == y_test ))
    # baseline = np.zeros(y_test.shape)
    # print("Baseline prediction accuracy :", np.mean( baseline == y_test ))



    return 0

main()