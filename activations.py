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
    # K.dot(K.pow(K.sqrt(K.update_add(K.dot(alpha, K.pow(x, 2)), 1.0)), -1), x)
    # print(tf.shape(x))
    return tf.where(K.greater_equal(x, K.zeros(shape=tf.shape(x))),
                    x,
                    x / K.sqrt(1 +(alpha*K.pow(x, 2))))

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
    return x / K.sqrt(1 +(alpha*K.pow(x, 2)))

def bentID(x):
    '''
    Bent Identity
    .. math::
        bentID(x)=x\\left (\\frac{1}{\sqrt{1+\\alpha x^{2}}} \\right )
    '''

    return ((K.sqrt(K.pow(x,2)+1)-1)/2)+x
