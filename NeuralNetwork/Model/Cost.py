
from enum import Enum
import numpy as np

class Cost(Enum):
    SQUARE_RESIDUALS = 0
    CROSS_ENTROPY = 1

def SSR(X, Y, evaluate):
     sum = 0
     for i in range( len(X) ):
            predicted = evaluate(X[i])
            for j in range( predicted.shape[1] ):
                sum += ( Y[i] - predicted ) ** 2
     return sum

def SSR_Derivative(X, Y, data_index, output_values):
    matrix = np.zeros( (len(output_values), 1), dtype=np.float64 )
    for a in range( len(output_values) ):
            matrix[a,0] = -2 * ( Y[data_index] - output_values[a,0] )
    return matrix

def Cross_Entropy(X, Y, evaluate):
       sum = 0
       for a in range( len(X) ):
           predicted = evaluate(X[a])
           index = np.where( max(predicted) == predicted )
           sum += -np.log(predicted[index])
       return sum

def Cross_Entropy_Derivative(X, Y, data_index, output_values):
       matrix = np.zeros( (len(output_values), 1), dtype=np.float64 )
       for a in range( len(output_values) ):
           index = np.where( max(output_values) == output_values )
           matrix[a,0] = -1 / output_values[index,0]
       return matrix