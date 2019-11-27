from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

import tensorflow as tf

def ISRLU(x, alpha=1.0):
    '''
    Applies the ISRLU function element-wise:
    .. math::
        ISRLU(x)=\\left\\{\\begin{matrix} x, x\\geq 0 \\\\  x * (\\frac{1}{\\sqrt{1 + \\alpha*x^2}}), x <0 \\end{matrix}\\right.
    Plot:
    .. figure::  _static/isrlu.png
        :align:   center
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Arguments:
        - alpha: hyperparameter α controls the value to which an ISRLU saturates for negative inputs (default = 1)
    References:
        - ISRLU paper: https://arxiv.org/pdf/1710.09967.pdf
    '''
    return tf.where(K.greater_equal(x, 0),
                    x,
                    K.dot(K.pow(K.sqrt(K.update_add(K.dot(alpha, K.pow(x, 2.0)), 1.0)), -1), x))

def ISRU(x, alpha=1.0):
    '''
    Applies the ISRU function element-wise:
    .. math::
        ISRU(x)=x\\left (\\frac{1}{\sqrt{1+\\alpha x^{2}}} \\right )
    Plot:
    .. figure::  _static/isrlu.png
        :align:   center
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Arguments:
        - alpha: hyperparameter α controls the value to which an ISRLU saturates for negative inputs (default = 1)
    References:
        - ISRLU paper: https://arxiv.org/pdf/1710.09967.pdf
    '''
    return K.dot(K.pow(K.sqrt(K.update_add(K.dot(alpha, K.pow(x, 2.0)), 1.0)), -1), x)