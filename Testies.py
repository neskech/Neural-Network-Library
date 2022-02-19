
from turtle import back
import numpy as np

shape = (2,2)
inputShape = (1,4,4)
output_shape = (1,3,3)
kernel_shape = (1,2,2)
kernels = np.ones((1,2,2))
print(kernels)
stride = 1
input_shape = (1,4,4)
def avgPool( inputs):
        
        #Inputs is a 3D image with 1 or more depth dimensions that we must pool 
        output = np.zeros( output_shape )
        #Loop through each image depth layer
        for s in range( output_shape[0] ):
            for a in range(0, inputShape[1] - shape[0] + 1, stride):
                 for b in range(0, inputShape[2] - shape[1] + 1, stride):
                         row_extent = a + shape[0]
                         col_extent = b + shape[1]
                         
                         if row_extent > inputShape[1] or col_extent > inputShape[2]: 
                             print('YEYEYEYE')
                             break
                         
                         print(a, ' ', b, ' ', a + shape[0], ' ', b + shape[1])
                         output[s,a,b] =  np.sum( inputs[s, a : a + shape[0], b : b + shape[1]] ) / (shape[0] * shape[1])
  
             
        return output
  
def conv( inputs):
        #Convolution operation
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        output = np.zeros( output_shape )
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( kernel_shape[0] ):
              for a in range(0, output_shape[1], stride):
                 for b in range(0, output_shape[2], stride):
                         row_extent = a + kernel_shape[1]
                         col_extent = b + kernel_shape[2]
                         
                         if row_extent > inputShape[1] or col_extent > inputShape[2]: 
                             break
                          
                         print('SUB ARRAY \n',inputs[ :, a : a + kernel_shape[1], b : b + kernel_shape[2] ] )
                         print('KERNEL \n', kernels[depth, :, :])
                         print('RESULT \n',np.sum(np.multiply( inputs[ :, a : a + kernel_shape[1], b : b + kernel_shape[2] ], kernels[depth, :, :] ) ) ) 
                         #print( np.dot( inputs[ :, a : a + kernel_shape[1], b : b + kernel_shape[2] ], kernels[depth, :, :] ) )
             
        return output
    
def maxPool( inputs):
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        output = np.zeros( output_shape )
        for depth in range(0, output_shape[0], stride):
           for a in range(0, output_shape[1], stride):
                for b in range(0, output_shape[2], stride):
                         row_extent = a + shape[0]
                         col_extent = b + shape[1]
                         
                         if row_extent > inputShape[1] or col_extent > inputShape[2]: 
                             break
                         
                         output[depth,a,b] =  np.max( inputs[depth, a : a + shape[0], b : b + shape[1] ] )
  
  
             
        return output
    
def derive( dLdZ, X):
         #kernelShape[0] = numKernels
         size = kernel_shape[0]
         Dbiases = np.zeros(size)
         Dkernels = np.zeros( kernel_shape )
         
         #Size is the number of kernels
         for depth in range(size):
              for a in range(X.shape[1]):
                 for b in range(X.shape[2]):
                         row_extent = a + dLdZ.shape[1]
                         col_extent = b + dLdZ.shape[2]
                         
                         if row_extent > X.shape[1] or col_extent > X.shape[2]: 
                             break
                         
                         Dkernels[depth,a,b] = np.sum( np.multiply( X[ :, a : a + dLdZ.shape[1], b : b + dLdZ.shape[2] ], dLdZ )  )
                         
              Dbiases[depth] = np.sum(dLdZ[depth, :, :])
           
         #Sum up
         return Dkernels, Dbiases 
     
def back_process( inputs):
        #Input is our dL/dZ. This convolotuion gets us dL/dK
        #Reverse the shape of the output back into the input image shape X     
        if len(inputs.shape) == 2:
            inputs = inputs[np.newaxis, ...]
            
        inputs = np.pad(inputs, pad_width=1)           
        output = np.zeros( input_shape )
        #Loop for each kernel, no need to loop through the third dimension of the kernels since its constant
        for depth in range( input_shape[0] ):
              for a in range(0, input_shape[1], stride):
                 for b in range(0, input_shape[2], stride):
                         row_extent = a + kernel_shape[1]
                         col_extent = b + kernel_shape[2]
                         
                         if row_extent > inputs.shape[1] or col_extent > inputs.shape[2]: 
                             break
                         
                         output[depth,a,b] = np.sum(np.multiply( inputs[ :, a : a + kernel_shape[1], b : b + kernel_shape[2] ], kernels[depth, :, :] ) )
        return output
    
     
input = [ [2,12,4], [3,2,5], [1,2,3]  ]
input = np.array(input)
print(back_process(input))