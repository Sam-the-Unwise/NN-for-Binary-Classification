# NN-for-Binary-Classification
A python script that creates a neural network for binary classification


# Gradient-Descent
Here is a link to our NNOneSplit function for convenience: https://github.com/Sam-the-Unwise/NN-for-Binary-Classification/blob/master/stochastic_gradient.py


## BROAD SUMMARY
This NNOneSplit function works by creating a v_mat (matrix) of random numbers close to 0 as well as a w_vec (vector) of random numbers close to 0. An X_train, y_train, X_validation, and y_validation sets are then created using the inputted X_mat and y_vec. The function then goes through a loop up until the max_epochs that are inputted into the function. 

Within the function, a loop over the amount of epochs and an inner loop over the current observation in the X_train matrix. Within the inner loop, y_hat and w_vec are calculated by using a forward application approach. Outside the inner loop but inside the outer loop, the logistic loss of the y_train and y_validation are calculated and appended to arrays that will be returned at the end of the function  in the form of a single double-array with y_train_predictions as the first subarray and the y_val_predictions as the second subarray. At the end of the function, the return values will be [v_mat, w_vec, loss_values]. Please see the section <b>HOW TO RUN THIS FUNCTION</b> for further descriptions on what each of these values holds.


## INDEPTH SUMMARY
This NNOneSplit function works by creating a v_mat (matrix) of random numbers close to 0 as well as a w_vec (vector) of random numbers close to 0. An X_train, y_train, X_validation, and y_validation sets are then created using the inputted X_mat and y_vec. The function then goes through a loop up until the max_epochs that are inputted into the function. 

Within this outer loop is a second loop that will loop over each observation in our X_train matrix. Within this loop the y_tild is determine based on the corresponding element (to the current X_train observation) in the y_train vector. h_v is then calculated by determining the sigmoid of the transpose of v_mat matrix multiplied by the current corresponding element of our X_train. This h_v is then used to determine the first, second, and third term of our graident function. The gradient function's output is used to calculate the v_mat by multiplying the gradient by the step_size and subtracting that result from the original v_mat.

Next, still within the second for loop, the function will calculate y_hat. This is done by taking the sigmoid of w_vec matrix multiplied by h_v. This y_hat is then used to calculate the new first, second, and third terms of our gradient function. The output of this gradient function is used to calculate the w_vec but multiplying the gradient by the step_size and subtracting that result from the original w_vec.

Breaking out of the inner for loop, the outer loop then calculates the y_train_pred (logistic loss prediction) by calculating the sigmoid of the X_train matrix_multiplied by the v_mat, then matrix multiplying the result by the w_vec and taking the sigmoid of that. This is appending the log_loss of y_train and y_train_pred to a loss_train matrix. y_val_pred is calculated in the same way as y_train_pred , except X_validation is used instead of X_train. loss_val is then appended with the log_loss of the y_validation and the y_val_pred.

Finally, outside the last for loop, a loss_values double matrix is created where the loss values from loss_train are saved in the first slot while the loss values from loss_val are saved in the second slot. At the end of this entire function, the return values are [v_mat, w_vec, and loss_values]




## HOW TO RUN THIS FUNCTION
To run this function, you must input all of the following arguements into the function call, with the function call being formatted as: NNOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_train)

Below are the arguments and the expected returns (return items will be returned in list format)

INPUT ARGS:
       X_mat : feature matrix (n_observations x n_features)
       y_vec : label vector (n_observations x 1)
       max_epochs : scalar int > 1
       step_size
       n_hidden_units : number of hidden units
       is_train : logical vector of size n_observations
           - TRUE if the observation is the train set
           - FALSE for the validation set
Return: list/dictionary containing at least three elements named:
           - v_mat : best weight matrix (n_features x n_hidden_units) used to
                          predict hidden units given inputs
           - w_vec : best weight vector (n_hidden_units) used to predict output
                          given hidden units
           - loss_values : a matrix/data_table/etc which stores the logistic
                          loss with respect to the train/validation set for each
                          iteration
