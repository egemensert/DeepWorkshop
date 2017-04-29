# METU CClub Deep Learning Workshop 2017
# Heavily based on CS231n Assignment 2
# This script holds the essential forward and backward pass functions.

import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass of the fully connected layer.

    N is the number of samples.
    D is the number of dimensions.

    Inputs:
    - x: A numpy array of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b) -for backpropagation-
    """
    out = None
    ########################################################################
    # TODO: Calculate {out}                                                #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass of the fully connected layer.

    Inputs:
    - dout: Error input, of shape (N, M)
    - cache: Tuple of:
        - x: Input data, of shape (N, D)
        - w: Weights, of shape (D, M)

    - Returns a tuple of:
        - dx: Gradient w.r.t. x, of shape (N, D)
        - dw: Gradient w.r.t. w, of shape (D, M)
        - db: Gradient w.r.t. b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Calculate {dw, dw, db}                                         #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out, cache = None, x
    ########################################################################
    # TODO: Calculate {out}                                                #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ########################################################################
    # TODO: Calculate {dx}                                                 #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return dx

def softmax_forward(x):
    out, cache = None, x
    ########################################################################
    # TODO: Calculate {out}                                                #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return out, cache

def softmax_backward(dout, cache):
    dx, x = None, cache
    ########################################################################
    # TODO: Calculate {dx}                                                 #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return dx

def mean_squared_loss(x, y):
    """
    Computes the mean squared loss function.

    C denotes the number of classes.

    Input:
        - x of shape (N, C)
        - y of shape (N, C)
    """
    loss = 0.
    dx = None
    ########################################################################
    # TODO: Calculate {dx, loss}                                           #
    ########################################################################

    ########################################################################
    # End of the code                                                      #
    ########################################################################
    return loss, dx
