###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Kruse
# DESCRIPTION: NNetOneSplit function used for stochastic gradient
# VERSION: 1.0.0v
#
###############################################################################


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
def NNOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_train):
    n_features = X_mat.shape[1]
    #print(n_features)

    # initialize v_mat and w_vec to some random number close to zero
    np.random.seed( 0 )
    v_mat = np.random.randn( n_features, n_hidden_units )
    w_vec = np.random.randn( n_hidden_units )
    #print(v_mat)
    #print(w_vec)

    #print(v_mat.shape[1])

    loss_train = []
    loss_val = []

    #   for is_train, randomly assign 60% train and 40% validation
    #X_train, X_validation, y_train, y_validation = split_matrix(X_mat, y_vec, .6)
    X_train = np.delete( X_mat, np.argwhere(is_train==True), 0)
    y_train = np.delete(y_vec, np.argwhere(is_train == True), 0)
    X_validation = np.delete(X_mat, np.argwhere(is_train == False), 0)
    y_validation = np.delete(y_vec, np.argwhere(is_train == False), 0)

    #is_train = np.array([])

    X_train_i = X_train[0]

    index=0
    #print( sigmoid( v_mat[ index ] * X_train[ index ] ) )

    #v_mat = v_mat - step_size * gradient

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
        y_train_pred = np.around(sigmoid_2(sigmoid_2(X_train @ v_mat) @ w_vec ))
        #print(y_train_pred)
        #print("hi", np.mean( y_train != y_train_pred ))
        #loss_train.append( np.mean( y_train != y_train_pred ) )
        loss_train.append( log_loss( y_train, y_train_pred ))

        y_val_pred = np.around(sigmoid_2(sigmoid_2(X_validation @ v_mat) @ w_vec))
        #print(y_val_pred)
        #print("bye", np.mean( y_validation != y_val_pred))
        #loss_val.append( np.mean( y_validation != y_val_pred) )
        loss_val.append( log_loss( y_validation, y_val_pred ))

    loss_values = [[],[]]
    loss_values[0].append(loss_train)
    loss_values[1].append(loss_val)
    #print(loss_values)

    return v_mat, w_vec, loss_values