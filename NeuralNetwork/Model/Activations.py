import numpy as np
from enum import Enum

class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    
def Relu(input, predicted_index):
   matrix = np.zeros(input.shape)
   with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -300, 300)
            y[...] = max(0, val)
   return matrix

def softPlus(input, predicted_index):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = np.log( 1 + np.e ** val )
     return matrix       
 
def sigmoid(input, predicted_index):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = 1  / ( 1 + np.e ** -val )
     return matrix
 
def hyperbolic_tangent(input, predicted_index):
    matrix = np.zeros(input.shape)
    with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            val = val = np.clip(x, -15, 15)
            y[...] = ( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val)
    return matrix

def softMax(matrix, predicted_index):
    output = np.zeros(matrix.shape)
    
    denom = 0
    for a in range(matrix.shape[0]):
        denom += np.e ** matrix[a, 0]
        
    for a in range(matrix.shape[0]):
        numerator = np.e ** matrix[a,0]
        output[a,0] = numerator / denom
     
    return output

def argMax(input): 
    max_ = input[0, 0]
    maxidx
    for a in range(input.shape[0]):
        if input[a, 0] > max_:
            max_ = input[a, 0]
            maxidx = a
    return maxidx
    
            

def Relu_Deriv(input, predicted_index):
      matrix = np.zeros(input.shape)
      with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
            y[...] = np.clip(x, 0, 1)
      return matrix

def softPlus_Deriv(input, predicted_index):
     matrix = np.zeros(input.shape)
     with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = (np.e ** val) / ( 1 + np.e ** val )
     return matrix
 
def sigmoid_Deriv(input, predicted_index):
    matrix = np.zeros(input.shape)
    with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = ( np.e ** -val ) / ( ( 1 + np.e ** -val) ** 2 ) 
    return matrix

def hyperbolic_tangent_Deriv(input, predicted_index):
   matrix = np.zeros(input.shape)
   with np.nditer([input, matrix], op_flags=['readwrite']) as it:
        for x, y in it:
           val  = np.clip(x, -15, 15)
           y[...] = 1 - (( np.e ** val - np.e ** -val) / ( np.e ** val + np.e ** -val) ) ** 2
   return matrix

def softMax_Deriv(input, predicted_index):
    #input is not softmaxed
    input = softMax(input, predicted_index)
    matrix = np.zeros(input.shape)
    for a in range(input.shape[0]):
        if a == predicted_index:
            matrix[a,0] = input[predicted_index,0] * (1 - input[predicted_index,0])
        else:
            matrix[a,0] = -input[predicted_index,0] * input[a,0]
    return matrix
   