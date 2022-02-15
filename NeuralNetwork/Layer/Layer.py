
from enum import Enum
from tkinter.tix import MAX
import numpy as np


class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    
class POOL(Enum):
    MAX = 0
    AVG = 1
    NONE = 2

class Layer:
    def __init__(self, size : int, func : ACT_FUNC) -> None:
        self.size = size
        self.activation_func = func
    
    def process(self, inputs):
        pass
    
    def init_rand_params(self, mean : float = 0, SD : float = 1, seed : int):
        pass
    
    def activation(self, x : np.float64):
        if self.FUNC is ACT_FUNC.RELU:
            x = np.clip(x,-300,300)
            return max(0,x)

        elif self.FUNC is ACT_FUNC.SOFT_PLUS:
             x = np.clip(x,-15,20)
             return np.log( 1 + np.e ** x )

        elif self.FUNC is ACT_FUNC.SIGMOID:
            x = np.clip(x,-20,20)
            return  1  / ( 1 + np.e ** -x )
        elif self.FUNC is ACT_FUNC.TANH:
            x = np.clip(x,-10,10)
            return ( np.e ** x - np.e ** -x) / ( np.e ** x + np.e ** -x)
    
    def activation_deriv(self, x : np.float64):
        if self.FUNC is ACT_FUNC.RELU:
            return 1 if x >= 0 else 0

        elif self.FUNC is ACT_FUNC.SOFT_PLUS:
                x = np.clip(x,-15,20)
                return (np.e ** x) / ( 1 + np.e ** x )

        elif self.FUNC is ACT_FUNC.SIGMOID:
            x = np.clip(x,-20,20)
            return  ( np.e ** -x ) / ( ( 1 + np.e ** -x) ** 2 ) 
        
        elif self.FUNC is ACT_FUNC.TANH:
            x = np.clip(x,-10,10)
            return 1 - (( np.e ** x - np.e ** -x) / ( np.e ** x + np.e ** -x) ) ** 2

    


class ConvolutionLayer(Layer):
     def __init__(self, size : int, func : Layer.ACT_FUNC, kenerl_shape, 
                  kernel_stride, pool_stride, pool_type, input_shape : tuple[int,int,int] = (-1,-1,-1) ) -> None:    
        super().__init__(size, func)
        self.biases = np.zeros( shape= (self.size, 1) )
        
     def init_rand_params(self, mean : float = 0, SD : float = 1, seed : int):
         pass
     
     def set_input_shape(self, inputShape):
         self.inputShape = inputShape

class DenseLayer(Layer):
    def __init__(self, size : int, func : Layer.ACT_FUNC) -> None:
        super().__init__(size, func)
        self.biases = np.zeros( shape= (self.size, 1) )
    
    def init_weights(self, next_layer_size : int):
        self.weights = np.zeros( shape=(self.size, next_layer_size) )
        
    def init_rand_params(self, mean : float = 0, SD : float = 1, seed : int):
        pass