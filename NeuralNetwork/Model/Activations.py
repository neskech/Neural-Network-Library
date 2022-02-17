
import numpy as np
from enum import Enum

class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    
def Relu(matrix):
   with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            val = val = np.clip(x, -300, 300)
            x[...] = max(0, val)

def softPlus(matrix):
     with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            val = val = np.clip(x, -15, 15)
            x[...] = np.log( 1 + np.e ** val )

def sigmoid(matrix):
     with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            val = val = np.clip(x, -15, 15)
            x[...] = 1  / ( 1 + np.e ** -val )

def hyperbolic_tangent(matrix):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            val = val = np.clip(x, -15, 15)
            x[...] = ( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val)

def softMax(matrix):
    output = np.zeros( matrix.shape )
    for a in range( matrix.shape[0] ):
        numerator = np.e ** matrix[a,0]
        denom = 0
        for b in range( matrix.shape[0] ):
            denom += np.e ** matrix[b,0]
        output[a,0] = numerator / denom
        
    return output
            

def Relu_Deriv(matrix):
      with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.clip(x, 0, 1)


def softPlus_Deriv(matrix):
     with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
           val  = np.clip(x, -15, 15)
           x[...] = (np.e ** val) / ( 1 + np.e ** val )

def sigmoid_Deriv(matrix):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
           val  = np.clip(x, -15, 15)
           x[...] = ( np.e ** -val ) / ( ( 1 + np.e ** -val) ** 2 ) 

def hyperbolic_tangent_Deriv(matrix):
   with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
           val  = np.clip(x, -15, 15)
           x[...] = 1 - (( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val) ) ** 2

def softMax_Deriv(matrix):
    pass

