
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
        self.setActivation(func)
    
    def process(self, inputs):
        pass
    
    def set_input_size(self, layer):
        pass
    
    def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        pass
    
    def back_process(self, inputs):
        pass
    
    def activate(self, inputs, predicted_index = -1, use_derivative : bool = False):
         if self.activation is None and use_derivative == False:
             return inputs
         if self.activation_derivative is None and use_derivative:
             return None
         elif use_derivative and predicted_index == -1:
             raise Exception('ERROR: You\'re using the derivative but have a pred index of -1')
         if not use_derivative:
             return self.activation(inputs, predicted_index)
         else:
             return self.activation_derivative(inputs, predicted_index)
         
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
     def __init__(self, num_kernels : int, func : ACT_FUNC, kernel_shape, input_shape : tuple[int,int,int] = None, 
                  stride : int = 1, ) -> None:    
        super().__init__(num_kernels, func)
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.stride = stride

        if not input_shape is None:
          self.kernel_shape = (input_shape[0], self.kernel_shape[0], self.kernel_shape[1])
          self.kernels = np.zeros(shape = (self.size,) + self.kernel_shape, dtype=np.float64)
          #3D Output. 3rd dimension is for the amount of kernels we have, the 2D is the result of one kernel convolution
          self.output_shape = (self.size, int( (self.input_shape[1] - self.kernel_shape[1]) / self.stride + 1 ), int( (self.input_shape[2] - self.kernel_shape[2]) / self.stride + 1 ) )
          self.biases = np.zeros(shape = (self.size,1), dtype=np.float64)
          print('Conv output shape',self.output_shape)
            
        
     def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
         if seed != -1:
             np.random.RandomState(seed)
            
         with np.nditer( self.kernels, op_flags=['readwrite']) as it:
             for x in it:
                 x[...] = np.random.normal(mean, SD)
   
     
     def process(self, inputs):
        #Convolution operation
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
        if inputs.shape != self.input_shape:
            raise Exception(f'ERROR: In convolution.process() - Input array shape {inputs.shape} does not match required input shape of {self.input_shape}')
           
        #print(self.output_shape) 
        output = np.zeros( self.output_shape )
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( self.size ):
              for a in range(0, self.output_shape[1], self.stride):
                 for b in range(0, self.output_shape[2], self.stride):
                         row_extent = a + self.kernel_shape[1]
                         col_extent = b + self.kernel_shape[2]
                         
                         if row_extent > self.input_shape[1] or col_extent > self.input_shape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + self.kernel_shape[1], b : b + self.kernel_shape[2] ], self.kernels[depth, :, :, :] ) )
             
        return output
      
    
     def back_process(self, inputs):
        #Input is our dL/dZ. This convolotuion gets us dL/dK
        #Reverse the shape of the output back into the input image shape X    
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
        if inputs.shape != self.input_shape:
            raise Exception(f'ERROR: In convolution.back_process() - Input array shape {inputs.shape} does not match required input shape of {self.input_shape}') 
            
        inputs = np.pad(inputs, pad_width=1)           
        output = np.zeros( self.input_shape )
        #Axis 0 is the number of kernels
        rKernels = np.rot90(rKernels, axes=(1,2))
        rKernels = np.rot90(rKernels, axes=(1,2))
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( self.input_shape[0] ):
              for a in range(0, self.input_shape[1], self.stride):
                 for b in range(0, self.input_shape[2], self.stride):
                         row_extent = a + self.kernel_shape[1]
                         col_extent = b + self.kernel_shape[2]
                         
                         if row_extent > inputs.shape[1] or col_extent > inputs.shape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + self.kernel_shape[1], b : b + self.kernel_shape[2] ], self.kernels[depth, :, :, :] ) )
        return output
     
     def derive(self, dLdZ, X):
         #kernelShape[0] = numKernels
         if len(dLdZ.shape) == 2:
            dLdZ = dLdZ[np.newaxis, ...]
        
         if len(X.shape) == 2:
            dLdZ = X[np.newaxis, ...]
            
         Dbiases = np.zeros((self.size,1))
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
          self.input_shape = prevLayerOutput
          
          #First index of dimensions tuple is the highest dimension. So [0] = 3D dimension
          #The 3D dimension of the kernel must = the 3D dimension of the input image, where the 3rd dimension is either the channels or the number of kernels
          self.kernel_shape = (prevLayerOutput[0], self.kernel_shape[0], self.kernel_shape[1])
          self.kernels = np.zeros(shape = (self.size,) + self.kernel_shape, dtype=np.float64)
          #3D Output. 3rd dimension is for the amount of kernels we have, the 2D is the result of one kernel convolution
          self.output_shape = (self.size, int( (self.input_shape[1] - self.kernel_shape[1]) / self.stride + 1 ), int( (self.input_shape[2] - self.kernel_shape[2]) / self.stride + 1) )
          self.biases = np.zeros(shape = (self.size,1), dtype=np.float64)
        

class MaxPoolLayer(Layer):
    def __init__(self, shape, stride : int) -> None:
        super().__init__(None, None)
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
                         
                         if row_extent > self.input_shape[1] or col_extent > self.input_shape[2]: 
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
          self.input_shape = layer.output_shape
          #The 3rd dimension from the last layer is maintained
          self.size = layer.output_shape[0]
          self.output_shape = (layer.output_shape[0], int( (self.input_shape[1] - self.shape[0]) / self.stride + 1 ), int( (self.input_shape[2] - self.shape[1]) / self.stride + 1) )
        #  print('max pool input shape',self.input_shape)
        #  print('max pool output shape',self.output_shape)

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
          self.output_shape = (self.size, int( (self.inputShape[1] - self.shape[0]) / self.stride + 1 ), int( (self.inputShape[2] - self.shape[1]) / self.stride + 1) )

class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__(None, None)
       
    def process(self, inputs):
        return np.reshape(inputs, (-1,1) )
     
    def back_process(self, inputs):
        #Unflattens the array
       # print('input shape',inputs.shape)
        return np.reshape(inputs, self.input_shape)
    
     
    def set_input_size(self, layer : Layer):
        self.input_shape = layer.output_shape
        print('flatten input ',self.input_shape)
        self.output_shape = ( self.input_shape[0] * self.input_shape[1] * self.input_shape[2], )
        self.size = self.output_shape[0]
        
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
       # print('weights shape ',self.weights.shape, 'inputs shape ',inputs.shape)
        return np.matmul(self.weights.T, inputs)
             
    def set_input_size(self, layer : Layer):
        if self.weights is None or self.biases is None:
              self.biases = np.zeros( shape= (self.size, 1), dtype=np.float64 )
              self.weights = np.zeros( shape=(self.size, layer.size), dtype=np.float64 )
              self.output_shape = (self.size)
        