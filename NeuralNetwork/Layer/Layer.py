
from enum import Enum
from operator import mul
from NeuralNetwork.Model import Activations as acts
import numpy as np


class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    
class POOL(Enum):
    MAX = 0
    AVG = 1
    NONE = 2
    

class Layer:
    def __init__(self, size : int, func : ACT_FUNC) -> None:
        self.size = size
        self.activation_func = func
        if not func is None:
            self.setActivation(func)
    
    def process(self, inputs):
        pass
    
    def init(self, prevLayerShape, nextLayerSize):
        pass
    
    def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        pass
    
    def setActivation(self, func : ACT_FUNC):
        if func is ACT_FUNC.RELU:
            self.activation = acts.Relu
            self.activation_derivative = acts.Relu_Deriv

        elif func is ACT_FUNC.SOFT_PLUS:
            self.activation = acts.softPlus
            self.activation_derivative = acts.softPlus_Deriv

        elif func is ACT_FUNC.SIGMOID:
            self.activation = acts.sigmoid
            self.activation_derivative = acts.sigmoid_Deriv

        elif func is ACT_FUNC.TANH:
            self.activation = acts.hyperbolic_tangent
            self.activation_derivative = acts.hyperbolic_tangent_Deriv
            
        elif func is ACT_FUNC.SOFTMAX:
            self.activation = acts.softMax
            self.activation_derivative = acts.softMax_Deriv

class ConvolutionLayer(Layer):
     def __init__(self, num_kernels : int, func : Layer.ACT_FUNC, kernel_shape, input_shape : tuple[int,int,int] = (None,None,None), 
                  stride : int = 1, ) -> None:    
        super().__init__(num_kernels, func)
        self.kernel_shape = kernel_shape
        self.inputShape = input_shape
        self.stide = stride
        
     def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
         if seed != -1:
             np.random.RandomState(seed)
             
         for a in range(self.kernel_shape[0]):
             for b in range(self.kernel_shape[1]):
                 for c in range(self.kernel_shape[2]):
                     self.kernel[a,b,c] = np.random.normal(mean, SD)
     
     def process(self, inputs):
        #Convolution operation
        output = np.zeros( self.output_shape )
        for s in range( self.output_shape[2] ):
              for a in range(0, self.inputShape[0] - self.kernel_shape[0] + 1, self.stride):
                 for b in range(0, self.inputShape[1] - self.kernel_shape[1] + 1, self.stride):
                         row_extent = a + self.kernel_shape[0]
                         col_extent = b + self.kenerl_shape[1]
                         
                         if row_extent >= self.inputShape[0] or col_extent >= self.inputShape[1]: 
                             break
                         
                         output[s,a,b] = np.dot( inputs[ :, a : a + self.kernel_shape[0], b : b + self.kernel_shape[1] ], self.kenerl[s] ) 
                 else:
                      #If inner loop was not broken
                     continue
                 break    
             
        return output
                    
     def init(self, prevLayerShape, nextLayerSize):
          self.biases = np.zeros( shape= (self.size, 1) )
          if self.inputShape == (None,None,None): self.inputShape = prevLayerShape
          self.kernel_shape = (self.kernel_shape[0], self.kernel_shape[1], prevLayerShape[2])
          self.kenerl = np.zeros( self.kernel_shape )
          self.output_shape = (self.inputShape[0] - self.kernel_shape[0] + 1, self.inputShape[1] - self.kernel_shape[1] + 1, self.size)
        

class MaxPoolLayer(Layer):
    def __init__(self, size : int, shape, stride : int) -> None:
        super().__init__(size, None)
        self.shape = shape
        self.stride = stride
    
    def process(self, inputs):
        output = np.zeros( self.output_shape )
        for a in range(0, self.inputShape[0] - self.kernel_shape[0] + 1, self.stride):
             for b in range(0, self.inputShape[1] - self.kernel_shape[1] + 1, self.stride):
                         row_extent = a + self.kernel_shape[0]
                         col_extent = b + self.kenerl_shape[1]
                         
                         if row_extent >= self.inputShape[0] or col_extent >= self.inputShape[1]: 
                             break
                         
                         output[a,b] =  np.max( inputs[:, a : a + self.kernel_shape[0]] )
             else:
                  #If inner loop was not broken
                 continue
             break    
             
        return output
    
    def init(self, prevLayerShape, nextLayerSize):
          self.inputShape = prevLayerShape
          self.output_shape = (self.inputShape[0] - self.shape[0] + 1, self.inputShape[1] - self.shape[1] + 1, self.size)

class AvgPoolLayer(Layer):
    def __init__(self, size : int, shape, stride : int) -> None:
        super().__init__(size, None)
        self.shape = shape
        self.stride = stride
    
    def process(self, inputs):
        output = np.zeros( self.output_shape )
        for a in range(0, self.inputShape[0] - self.kernel_shape[0] + 1, self.stride):
             for b in range(0, self.inputShape[1] - self.kernel_shape[1] + 1, self.stride):
                         row_extent = a + self.kernel_shape[0]
                         col_extent = b + self.kenerl_shape[1]
                         
                         if row_extent >= self.inputShape[0] or col_extent >= self.inputShape[1]: 
                             break
                         
                         output[a,b] =  np.sum( inputs[:, a : a + self.kernel_shape[0]] ) / (self.shape[0] * self.shape[1])
             else:
                  #If inner loop was not broken
                 continue
             break    
             
        return output
    
    def init(self, prevLayerShape, nextLayerSize):
          self.inputShape = prevLayerShape
          self.output_shape = (self.inputShape[0] - self.shape[0] + 1, self.inputShape[1] - self.shape[1] + 1, self.size)

class FlattenLayer(Layer):
    def __init__() -> None:
        super().__init__(None, None)
       
    def process(self, inputs):
         return np.ndarray.flatten(inputs)
     
    def init(self, prevLayerShape, nextLayerShape):
        self.inputShape = prevLayerShape
        self.outputShape = ( mul( self.inputShape ) )
        
class DenseLayer(Layer):
    def __init__(self, size : int, func : Layer.ACT_FUNC) -> None:
        super().__init__(size, func)
    
    def set_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        if seed != -1:
             np.random.RandomState(seed)
             
        for a in range( self.weigths.shape[0] ):
             for b in range( self.weights.shape[1] ):
                     self.weights[a,b] = np.random.normal(mean, SD)
                     
    def process(self, inputs):
        return np.matmul(self.weights, inputs) + self.biases
    
    def init(self, prevLayerShape, nextLayerSize):
        self.biases = np.zeros( shape= (self.size, 1) )
        self.weights = np.zeros(shape=(self.size, nextLayerSize))
        self.output_shape = nextLayerSize
        self.input_shape = prevLayerShape
        