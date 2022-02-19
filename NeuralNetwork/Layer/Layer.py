
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
    NONE = 5
    

class Layer:
    def __init__(self, size : int, func : ACT_FUNC) -> None:
        self.size = size
        self.activation_func = func
        self.output_shape = None
        self.input_shape = None
        
        if not func is None:
            self.setActivation(func)
    
    def process(self, inputs):
        pass
    
    def set_input_size(self, layer):
        pass
    
    def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        pass
    
    def back_process(self, input):
        pass
    
    def activate(self, inputs, use_derivative : bool):
         if self.activation is None and use_derivative == False:
             return inputs
         if self.activation_derivative is None and use_derivative:
             return None
         if not use_derivative:
             return self.activation(inputs)
         else:
             return self.activation_derivative(inputs)
         
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
            
        else:
            self.activation = None
            self.activation_derivative = None

class ConvolutionLayer(Layer):
     def __init__(self, num_kernels : int, func : ACT_FUNC, kernel_shape, input_shape : tuple[int,int,int] = (None,None,None), 
                  stride : int = 1, ) -> None:    
        super().__init__(num_kernels, func)
        self.kernel_shape = kernel_shape
        self.inputShape = input_shape
        self.stride = stride
        
     def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
         if seed != -1:
             np.random.RandomState(seed)
             
         for a in range(self.kernel_shape[0]):
             for b in range(self.kernel_shape[1]):
                 for c in range(self.kernel_shape[2]):
                     self.kernel[a,b,c] = np.random.normal(mean, SD)
     
     def process(self, inputs):
        #Convolution operation
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        output = np.zeros( self.output_shape )
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( self.kernel_shape[0] ):
              for a in range(0, self.output_shape[1], self.stride):
                 for b in range(0, self.output_shape[2], self.stride):
                         row_extent = a + self.kernel_shape[1]
                         col_extent = b + self.kernel_shape[2]
                         
                         if row_extent > self.inputShape[1] or col_extent > self.inputShape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + self.kernel_shape[1], b : b + self.kernel_shape[2] ], self.kernels[depth, :, :] ) )
             
        return output
      
    
     def back_process(self, inputs):
        #Input is our dL/dZ. This convolotuion gets us dL/dK
        #Reverse the shape of the output back into the input image shape X     
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        inputs = np.pad(inputs, pad_width=1)           
        output = np.zeros( self.input_shape )
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( self.input_shape[0] ):
              for a in range(0, self.input_shape[1], self.stride):
                 for b in range(0, self.input_shape[2], self.stride):
                         row_extent = a + self.kernel_shape[1]
                         col_extent = b + self.kernel_shape[2]
                         
                         if row_extent > inputs.shape[1] or col_extent > inputs.shape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + self.kernel_shape[1], b : b + self.kernel_shape[2] ], self.kernels[depth, :, :] ) )
        return output
     
     def derive(self, dLdZ, X):
         #kernelShape[0] = numKernels
         if len(dLdZ.shape) == 2:
            dLdZ = dLdZ[np.newaxis, ...]
            
         Dbiases = np.zeros(self.size)
         Dkernels = np.zeros( self.kernel_shape )
         output_shape = self.kernel_shape
         
         #Size is the number of kernels
         for depth in range(output_shape[0]):
              for a in range(output_shape[1]):
                 for b in range(output_shape[2]):
                         row_extent = a + dLdZ.shape[1]
                         col_extent = b + dLdZ.shape[2]
                         
                         if row_extent > X.shape[1] or col_extent > X.shape[2]: 
                             break
                         
                         Dkernels[depth,a,b] = np.sum( np.multiply( X[ :, a : a + dLdZ.shape[1], b : b + dLdZ.shape[2] ], dLdZ[depth, :, :] )  )
                         
              Dbiases[depth] = np.sum(dLdZ[depth, :, :])
           
         #Sum up
         return Dkernels, Dbiases
         
                   
     def set_input_size(self, layer : Layer):
          prevLayerOutput = layer.output_shape
          if self.inputShape == (None,None,None): self.inputShape = prevLayerOutput
          
          #First index of dimensions tuple is the highest dimension. So [0] = 3D dimension
          self.kernel_shape = (prevLayerOutput[0], self.kernel_shape[0], self.kernel_shape[1])
          self.kernels = np.zeros(shape = self.kernel_shape, dtype=np.float64)
          self.output_shape = (self.size, (self.inputShape[1] - self.kernel_shape[1]) / self.stride + 1, (self.inputShape[2] - self.kernel_shape[2]) / self.stride + 1)
          self.biases = np.zeros(shape = (self.size,1), dtype=np.float64)
        

class MaxPoolLayer(Layer):
    def __init__(self, size : int, shape, stride : int) -> None:
        super().__init__(size, None)
        self.shape = shape
        self.stride = stride
    
    def process(self, inputs):
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        output = np.zeros( self.output_shape )
        for depth in range(0, self.output_shape[0], self.stride):
           for a in range(0, self.output_shape[1], self.stride):
                for b in range(0, self.output_shape[2], self.stride):
                         row_extent = a + self.shape[0]
                         col_extent = b + self.shape[1]
                         
                         if row_extent > self.inputShape[1] or col_extent > self.inputShape[2]: 
                             break
                         
                         output[depth,a,b] =  np.max( inputs[depth, a : a + self.shape[0], b : b + self.shape[1] ] )
  
             
        return output
    
    def back_process(self, input):
        #Expanding input to its original size before it was shrinked by the pooling
        output = np.zeros( (input.shape[0], input.shape[1] * self.shape[0], input.shape[2] * self.shape[1]) )
        
        for depth in range(output.shape[0]):
           for a in range(0, output.shape[1], self.shape[0] ):
              for b in range(0, output.shape[2], self.shape[1] ):
                
                    val = input[depth, a // self.shape[0], b // self.shape[1]] 
                    subSection = input[depth, a : a + self.shape[0] , b : b + self.shape[1]]
                    max_index = np.where( subSection == max(subSection) )
                    
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                            if max_index == (i,j):
                                output[depth. a + i, b + j] = val
                            else:
                                output[depth, a + i, b + j] = 0
        return output
    
    def set_input_size(self, layer : Layer):
          self.inputShape = layer.output_shape
          self.output_shape = (self.size, (self.inputShape[0] - self.shape[0]) / self.stride + 1, (self.inputShape[1] - self.shape[1]) / self.stride + 1)

class AvgPoolLayer(Layer):
    def __init__(self, shape, stride : int) -> None:
        super().__init__(None, None)
        #Pooling is inherintely a 2D operation unlike convolutions
        self.shape = shape
        self.stride = stride
    
    def process(self, inputs):
        #Inputs is a 3D image with 1 or more depth dimensions that we must pool 
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
        output = np.zeros( self.output_shape )
        #Loop through each image depth layer
        for depth in range( self.output_shape[0] ):
            for a in range(0, self.output_shape[1], self.stride):
                 for b in range(0, self.output_shape[2], self.stride):
                         row_extent = a + self.shape[0]
                         col_extent = b + self.shape[1]
                         
                         if row_extent > self.inputShape[1] or col_extent > self.inputShape[2]: 
                             break
                         
                         output[depth,a,b] =  np.sum( inputs[depth, a : a + self.shape[0], b : b + self.shape[1]] ) / (self.shape[0] * self.shape[1])
  
             
        return output
    
    def back_process(self, input):
        #Expanding input to its original size before it was shrinked by the pooling
        output = np.zeros( (input.shape[0], input.shape[1] * self.shape[0], input.shape[2] * self.shape[1]) )
        
        for depth in range(output.shape[0]):
           for a in range(0, output.shape[1], self.shape[0] ):
              for b in range(0, output.shape[2], self.shape[1] ):
                
                    val = input[depth, a // self.shape[0], b // self.shape[1]] / (self.shape[0] * self.shape[1])
                    
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                                output[depth. a + i, b + j] = val
        return output
    
    def set_input_size(self, layer : Layer):
          self.inputShape = layer.output_shape
          #Size represents how many images we're pooling which corresponds to the 3D dimension of the input image
          self.size = layer.output_shape[0]
          self.output_shape = (self.size, (self.inputShape[0] - self.shape[0]) / self.stride + 1, (self.inputShape[1] - self.shape[1]) / self.stride + 1)

class FlattenLayer(Layer):
    def __init__() -> None:
        super().__init__(None, None)
       
    def process(self, inputs):
         return np.ndarray.flatten(inputs)
     
    def back_process(self, inputs):
        #Unflattens the array
        return np.reshape(inputs, self.input_shape)
    
     
    def set_input_size(self, prevLayerShape):
        self.inputShape = prevLayerShape
        self.outputShape = ( mul( self.inputShape ) )
        
class DenseLayer(Layer):
    def __init__(self, size : int, func : ACT_FUNC = ACT_FUNC.NONE, weights = None, biases = None) -> None:
        super().__init__(size, func)
        self.weights = weights
        self.biases = biases
    
    def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        if seed != -1:
             np.random.RandomState(seed)
        if self.weights is None or self.biases is None:
            return
             
        for a in range( self.weights.shape[0] ):
             for b in range( self.weights.shape[1] ):
                     self.weights[a,b] = np.random.normal(mean, SD)
                     
    def process(self, inputs):
       # print(f'WEIGHTS SHAPE {self.weights.shape} BIASES SHAPE {self.biases.shape} INPUTS SHAPE {inputs.shape}')
        return np.matmul(self.weights, inputs) + self.biases
    
    def back_process(self, inputs):
        #Unflattens the array
        return np.matmul(self.weights.T, inputs)
             
    def set_input_size(self, layer : Layer):
        if self.weights is None or self.biases is None:
              self.biases = np.zeros( shape= (self.size, 1), dtype=np.float64 )
              self.weights = np.zeros( shape=(self.size, layer.size), dtype=np.float64 )
              self.output_shape = (self.size)
        