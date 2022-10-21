from Layer import Layer, ACT_FUNC
import numpy as np

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
      
    
     def back_process(self, inputs, inputs_two):
        #Input is our dL/dZ. This convolotuion gets us dL/dK
        #Reverse the shape of the output back into the input image shape X    
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
        if inputs.shape != self.output_shape:
            raise Exception(f'ERROR: In convolution.back_process() - Input array shape {inputs.shape} does not match required input shape of {self.input_shape}') 
           
       # print('BACK PROCESS  INPUTS SHAPE', inputs.shape) 
        inputs = np.pad(inputs,pad_width=( (0, 0), (1,1), (1,1) ))      
        #print('BACK PROCESS  INPUTS SHAPE PADDING', inputs.shape)      
        output = np.zeros( self.input_shape )
        #Axis 0 is the number of kernels
        rKernels = np.rot90(self.kernels, axes=(1,2))
        rKernels = np.rot90(rKernels, axes=(1,2))
       # print(rKernels.shape, 'KENRNENENEL',self.kernels.shape)
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( self.input_shape[0] ):
              for a in range(0, self.input_shape[1], self.stride):
                 for b in range(0, self.input_shape[2], self.stride):
                         row_extent = a + self.kernel_shape[1]
                         col_extent = b + self.kernel_shape[2]
                         
                         if row_extent > inputs.shape[1] or col_extent > inputs.shape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + self.kernel_shape[1], b : b + self.kernel_shape[2] ], rKernels[depth, :, :, :] ) )
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
        
