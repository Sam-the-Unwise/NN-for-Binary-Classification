###############################################################################
#
# AUTHOR(S): Samantha Muellner
# DESCRIPTION: program that will implement a stochastic gradient descent algo
#       for a neural network with one hidden layer
# VERSION: 1.0.0v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss
from sklearn import neighbors, datasets
import random

# global variables
MAX_ITERATIONS = 20
STEP_SIZE = .1
N_HIDDEN_UNITS = 1



# Function: NNetOneSplit
# INPUT ARGS:
#       X_mat : feature matrix (n_observations x n_features)
#       y_vec : label vector (n_observations x 1)
#       max_iterations : scalar int > 1
#       step_size
#       n_hidden_units : number of hidden units
#       is_train : logical vector of size n_observations
#           - TRUE if the observation is the train set
#           - FALSE for the validatoin set
# Return: list/dictionary containing at least three elements named:
#           - v_mat : best weight matrix (n_features x n_hidden_units) used to
#                          predict hidden units given inputs
#           - w_vec : best weight vector (n_hidden_units) used to predict output
#                          given hidden units
#           - loss_values : a martix/data_table/etc which stores the logistic 
#                          loss with respect to the train/validation set for each
#                          iteration
def NNetOneSplit(X_mat, y_vec, max_iterations, step_size, n_hidden_units, is_train):
    # initialize v_mat and w_vec to some random number close to zero
    v_mat = np.array([])
    w_vec = np.array([])
    loss_values = np.array([])

    #   for is_train, randomly assign 60% train and 40% validation
    X_train, X_validation, y_train, y_validation = split_matrix(X_sc, y_vec)

    # during each iteration compute the gradients of v_mat and w_vec
    #       by taking a step (scaled by step_size) in the neg. gradient direction-
    for num in range(1, max_iterations):
        i = 0 # stand in variables please delete 0.0

        # at each iteration compute the log. loss on the train/validation sets
        #       store in loss_values

    
    return v_mat, w_vec, loss_values


# Function: split matrix
# INPUT ARGS:
#   X_mat : matrix to be split
#   y_vec : corresponding vector to X_mat
# Return: train, validation, test
def split_matrix(X_mat, y_vec):
    # split data 60% train by 40% validation
    X_train, X_validation = np.split( X_mat, [int(.6 * len(X_mat))])
    y_train, y_validation = np.split( y_vec, [int(.6 * len(y_vec))])

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


# Function: main
def main():
    # use spam data set
    data_matrix_full = convert_data_to_matrix("spam.data")
    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    y_vec = data_matrix_full[:,57]

    X_sc = scale(X_Mat)

    # logical vector of size n_observations
    # TRUE if the observation is in thet train set
    # FALSE for the validation set
    is_train = np.array([])
    


    # use NNetOneSplit with whole dataset for X_mat/y_vec
    NNetOneSplit(X_sc, y_vec, MAX_ITERATIONS, STEP_SIZE, N_HIDDEN_UNITS, is_train)


    ########### TO DO #############
    # plot the train/validation loss as a function of the number of iterations
    #   and draw a point to emphasize the minimu0m of each curve
    
    # the train loss should always go down, whereas the validation loss shoudl go 
    #   down and then start going up after a certain number of iterations
    #       if it does not, try decreasing the step_size and increasing max_iterations

    # Use 4-fold cross fold validation to compare the prediction accuracy of two algorithms
    #   (1) baseline/underfit - predict the most frequent classs
    #   (2) NNetOneSplit
    #   - plot the resulting test accuracy values as a function of the data set


    ########### EXTRA CREDIT #########
    # if you show gradient descent (from project 1, logistic regression with num iterations
    #   selected by a held-out validation set) in your test accuracy figure
    #   as a baseline

    # if you show NearestNeighborsCV (from project 2) in your test accuracy figure as a
    #   baseline

    # if you compute and plot ROC curves for each (test fold algorithm) combination
    #   make sure each algorithm is drawn in a different color, and there is a legend
    #   that the reader can use to read the figure

    # if you compute the area under the ROC curve (AUC) and include that as another 
    #   evaluation metric (in a separate panel/plot) to compare the test accruacy of the algo
    

    return 0