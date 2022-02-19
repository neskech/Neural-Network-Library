
import itertools
import numpy as np
from enum import Enum

class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    
def Relu(input):
   matrix = np.zeros(input.shape)
   with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -300, 300)
            y[...] = max(0, val)
   return matrix

def softPlus(input):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = np.log( 1 + np.e ** val )
     return matrix       
def sigmoid(input):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = 1  / ( 1 + np.e ** -val )
     return matrix
def hyperbolic_tangent(input):
    matrix = np.zeros(input.shape)
    with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = ( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val)
    return matrix
def softMax(matrix):
    output = np.zeros( matrix.shape )
    for a in range( matrix.shape[0] ):
        numerator = np.e ** matrix[a,0]
        denom = 0
        for b in range( matrix.shape[0] ):
            denom += np.e ** matrix[b,0]
        output[a,0] = numerator / denom
        
    return output
            

def Relu_Deriv(input):
      matrix = np.zeros(input.shape)
      with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            y[...] = np.clip(x, 0, 1)
      return matrix

def softPlus_Deriv(input):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = (np.e ** val) / ( 1 + np.e ** val )
     return matrix
def sigmoid_Deriv(input):
    matrix = np.zeros(input.shape)
    with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = ( np.e ** -val ) / ( ( 1 + np.e ** -val) ** 2 ) 
    return matrix
def hyperbolic_tangent_Deriv(input):
   matrix = np.zeros(input.shape)
   with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = 1 - (( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val) ) ** 2
   return matrix
def softMax_Deriv(input):
    pass

