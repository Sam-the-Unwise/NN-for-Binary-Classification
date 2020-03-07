###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Holguin
#            Jacob Christiansen
# DESCRIPTION: Gradient Descent Function
# VERSION: 1.0.0v
#
# Link to original function: 
# https://github.com/Sam-the-Unwise/Gradient-Descent/blob/master/gradientDescent.py
###############################################################################

import numpy as np
import csv
from math import sqrt

# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : # Function: calculate_gradient
# INPUT ARGS:
#   matrix : input matrix row with obs and features
#   y_tild : modified y val to calc gradient
#   step_size : step fir gradient
#
# Return: [none]
def calculate_gradient(x_row, y_tild, step_size, weight_vector_transpose):
    # calculate elements of the denominator
    verctor_mult = np.multiply(weight_vector_transpose, x_row)
    inner_exp = np.multiply(y_tild, verctor_mult)
    denom = 1 + np.exp(inner_exp)

    numerator = np.multiply(x_row, y_tild)

    # calculate gradient
    gradient = numerator/denom

    return gradient



# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : pos int that controls how many steps to take
# Return: weight_matrix
def gradientDescent(X, y, step_size, max_iterations):
    # tuple of array dim (row, col)
    arr_dim = X.shape

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations

    # variable that initiates to the weight vector
    weight_vector = np.zeros(X_arr_col)

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    # weight_matrix = np.array(np
    #                     .zeros(wm_total_entries)
    #                     .reshape(X_arr_col, max_iterations))

    array_of_zeros = []

    for i in range(X_arr_col):
        array_of_zeros.append(0)

    weight_matrix = np.array(array_of_zeros)

    # ALGORITHM
    weight_vector_transpose = np.transpose(weight_vector)

    for iteration in range(0, max_iterations):

        grad_log_losss = 0

        for index in range(0, X.shape[1]):
            #calculate y_tid
            y_tild = -1

            if(y[index] == 1):
                y_tild = 1


            grad_log_losss = 0
            verctor_mult = 0
            inner_exp = 0

            # variables for simplification
            gradient = calculate_gradient(X[index,:], 
                                            y_tild, 
                                            step_size, 
                                            weight_vector_transpose)

            grad_log_losss += gradient


        mean_grad_log_loss = grad_log_losss/X.shape[1]

        # update weight_vector depending on positive or negative
        weight_vector -= np.multiply(step_size, mean_grad_log_loss)

        # store the resulting weight_vector in the corresponding 
        #   column weight_matrix
        weight_matrix = np.vstack((weight_matrix, np.array(weight_vector)))

    # get rid of initial zeros matrix that was added
    weight_matrix = np.delete(weight_matrix, 0, 0)

    weight_matrix = np.transpose(weight_matrix)

    # end of algorithm
    return weight_matrix
