# Group Project # 2
# K-fold cross-validation for hyper-parameter tuning and model comparison
#
# Implementation of K-Fold cross-validation
#   Used it with machine learning algorithms:
#       (1) train hyper-parameters
#       (2) compare prediction accuracy.

# imports
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as skKnn
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from gradient_descent import gradient_descent

num_neigh = 5

def knn_wrap( X_train, y_train, X_new ) :
    num_neighbors = num_neigh
    lab_enc = preprocessing.LabelEncoder()
    Y_encoded = lab_enc.fit_transform(y_train)
    knn = skKnn(n_neighbors=num_neighbors)
    knn.fit(X_train, Y_encoded)
    return knn.predict_proba( X_new )

# Function: KFoldCV
# input arguments:
#   X_mat, a matrix of numeric inputs (one row for each observation, one column for each feature).
#   y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
#   ComputePredictions, a function that takes three inputs (X_train,y_train,X_new),
#                       trains a model using X_train,y_train,
#                       then outputs a vector of predictions (one element for every row of X_new).
#   fold_vec, a vector of integer fold ID numbers (from 1 to K).
def KFoldCV( X_mat, y_vec, ComputePredictions, fold_vec ):

    X_mat = np.array(X_mat)
    y_vec = np.array(y_vec)

    K = np.max(fold_vec)
    #K = fold_vec.shape[0]
    X_size = X_mat.shape[0]

    # The function should begin by initializing a variable called error_vec, a numeric vector of size K.
    error_vec = np.empty( K )
    new_y_mat = []
    new_pred_mat = []

    #id_vec = np.random.randint( 1, K+1, X_mat.shape[ 0 ] )

    # The function should have a for loop over the unique values k in fold_vec (should be from 1 to K).
    for value in range( 1, K+1 ) :
    #for fold_id in fold_vec :
        X_new = []
        Y_new = []
        X_train = []
        y_train = []

        # first define X_new,y_new based on the observations for which the corresponding elements of fold_vec
        #   are equal to the current fold ID k.
        for index in range(X_mat.shape[0]):
            if (value == fold_vec[index]):
                X_new.append(X_mat[index].tolist())
                Y_new.append(y_vec[index].tolist())
                #X_new = np.append(X_new, X_mat[index])
                #Y_new = np.append(Y_new, y_vec[index])

            # then define X_train,y_train using all the other observations
            else:
                X_train.append(X_mat[index].tolist())
                y_train.append(y_vec[index].tolist())
                #X_train = np.append(X_train, X_mat[index])
                #y_train = np.append(y_train, y_vec[index])

        # then call ComputePredictions
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_new = np.array(X_new)
        pred_new = ComputePredictions( X_train, y_train, X_new )

        if(pred_new.ndim > 1):
            pred_new = pred_new[:,1]

        new_y_mat.append( Y_new )
        new_pred_mat.append( pred_new )

        pred_new = np.around(pred_new)

        # then compute the zero-one loss of pred_new with respect to y_new
        #   and store the mean (error rate) in the corresponding entry of error_vec.
        error_vec[value-1] = np.mean(Y_new != pred_new)

    # At the end of the algorithm you should return error_vec.
    #print("omg", error_vec)
    return(error_vec, new_y_mat, new_pred_mat )

X_mat = np.array([[1,2,3],
                  [3,2,1],
                  [1,2,3],
                  [1,2,3],
                  [3,2,1],
                  [1,2,3]
                  ])
Y_vec = np.array([1,0,1,1,0,1])
#print(KFoldCV( X_mat, Y_vec, knn_wrap, fold_vec))

# function: NearestNeighborsCV
# input arguments
# X_mat, a matrix of numeric inputs/features (one row for each observation, one column for each feature).
# y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
# X_new, a matrix of numeric inputs/features for which we want to compute predictions.
# num_folds, default value 5.
# max_neighbors, default value 20.
def NearestNeighborsCV( X_mat, y_vec, X_new, num_folds=5, max_neighbors=20 ):
    global num_neigh
    # randomly create a variable called validation_fold_vec, a vector with integer values from 1 to num_folds.
    #validation_fold_vec = np.arange( 1, num_folds+1 )
    validation_fold_vec = np.random.randint(1, num_folds + 1, X_mat.shape[0])
    #print(validation_fold_vec)

    # initialize a variable called error_mat, a numeric matrix (num_folds x max_neighbors).
    error_mat = np.empty([num_folds, max_neighbors])

    # There should be a for loop over num_neighbors from 1 to max_neighbors.
    for index in range( 1, max_neighbors+1 ):

        # Inside the for loop you should call KFoldCV, and specify ComputePreditions=a function that uses your
        #   programming language’s default implementation of the nearest neighbors algorithm, with num_neighbors.
        #   e.g. scikit-learn neighbors in Python, class::knn in R.
        #   Store the resulting error rate vector in the corresponding column of error_mat.
        num_neigh = index
        error_col = KFoldCV( X_mat, y_vec, knn_wrap, validation_fold_vec )
        error_mat[:, index-1] = error_col[0][:]

    # Compute a variable called mean_error_vec (size max_neighbors) by taking the mean of each column of error_mat.
    mean_error_vec = np.empty(max_neighbors)
    for index in range(mean_error_vec.shape[0]):
        mean_error_vec[index] = np.mean( error_mat[:,index] )

    # Compute a variable called best_neighbors which is the number of neighbors with minimal error.
    best_neighbors = np.argmin(mean_error_vec)

    num_neigh = best_neighbors+1
    pred_new = knn_wrap(X_mat, y_vec, X_new )

    # Your function should output
    #   (1) the predictions for X_new, using the entire X_mat,y_vec with best_neighbors;
    #   (2) the mean_error_mat for visualizing the validation error.
    return(pred_new, mean_error_vec, error_mat, best_neighbors)

def NearestNeighborsCV_Wrapper( X_mat, y_vec, X_new ):
    results = NearestNeighborsCV( X_mat, y_vec, X_new )
    return results[0]
