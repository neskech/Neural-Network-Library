
from enum import Enum
from .Activations import argMax
import numpy as np

class Cost(Enum):
    SQUARE_RESIDUALS = 0
    CROSS_ENTROPY = 1

def Square_Residuals(X, Y, evaluate):
     sum = 0
     for i in range( len(X) ):
            predicted = evaluate(X[i])
            for a in range( predicted.shape[1] ):
                sum += ( Y[i, 0] - predicted[a, 0] ) ** 2
     return sum

def Square_Residuals_Derivative(X, Y, data_index, output_values):
    matrix = np.zeros( (len(output_values), 1), dtype=np.float64 )
    for a in range( len(output_values) ):
            matrix[a,0] = -2 * ( Y[data_index] - output_values[a,0] )
    return matrix

def Cross_Entropy(X, Y, evaluate):
       rand_data_points = np.random.randint(0, len(X), size=5)
       sum = 0
       for a in  rand_data_points :
           predicted = evaluate(X[a])
           index = Y[a]
           sum += -np.log(predicted[index])
       return sum

#TODO WTF IS THIS 
def Cross_Entropy_Derivative(X, Y, data_index, output_values):
       matrix = np.zeros((len(output_values), 1), dtype=np.float64)
       
       idx = None
       if len(Y.shape) > 1:
          idx = argMax(Y[data_index])
       else:
          idx = Y[data_index]
          
       for a in range( len(output_values) ):
           matrix[a,0] = -1 / (output_values[idx,0] + 1e-8)
       return matrix