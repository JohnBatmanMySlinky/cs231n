from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # for each training image
    for i in range(num_train):
        
        # scores is W*image
        # returns 10 scores, 1 for each class, for this given image we're on
        scores = X[i].dot(W)      
        
        # this is grabbing the score of the correct class for referene 
        correct_class_score = scores[y[i]]
        
        # so working with a vector that is one score per class
        for j in range(num_classes):
            # if the class is correct, skip cause no loss
            if j == y[i]:
                continue
            
            # this is the SVM loss
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            # this is the max(0,-)
            if margin > 0:
                loss += margin
            
                # from https://cs231n.github.io/optimization-1/
                # where the margin >0 the derivative = sum(x)
                # also noticing that the sign varies by if the class is correct
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # so now this should be 500 x 10, 
    # that is for each image we have 10 class predictions
    scores = X.dot(W)
    
    # this needs to be a vector of 500 x 1, the score for the correct class for each image
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train,1)
    
    # vectorized max
    margin = np.maximum(scores - correct_class_score + 1,0)
    
    # loss and dW where class is correct --> 0
    margin[np.arange(num_train),y] = 0
    
    # sum margin = loss
    loss = margin.sum()  
    
    # make an average
    loss /= num_train
    
    # add on regularization
    loss += reg * np.sum(W * W)
    
    # pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # margin but with indicator function
    margin_ind = margin
    margin_ind[margin_ind > 0] = 1
    
    margin_ind_sum = np.sum(margin_ind, axis = 1)
    margin_ind[np.arange(num_train),y] -= margin_ind_sum
    
    dW = (X.T).dot(margin_ind) / num_train + reg * 2 * W

    
    # pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
