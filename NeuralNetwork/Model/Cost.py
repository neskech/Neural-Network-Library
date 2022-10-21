
from enum import Enum
import numpy as np

class Cost(Enum):
    SQUARE_RESIDUALS = 0
    CROSS_ENTROPY = 1

def Square_Residuals(X, Y, evaluate):
     sum = 0
     for i in range( len(X) ):
            predicted = evaluate(X[i])
            for _ in range( predicted.shape[1] ):
                sum += ( Y[i] - predicted ) ** 2
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
       matrix = np.ones( (len(output_values), 1), dtype=np.float64 )
       return matrix
       for a in range( len(output_values) ):
           index = Y[data_index]
           matrix[a,0] = -1 / output_values[index,0]
       return matrix