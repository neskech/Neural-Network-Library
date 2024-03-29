from .Layer import Layer, parse_tuple
import numpy as np

class MaxPoolLayer(Layer):
    def __init__(self, shape = None, stride : int = None) -> None:
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
    
    def back_process(self, input, inputs_two):
        #Expanding input to its original size before it was shrinked by the pooling
        output = np.zeros( (input.shape[0], inputs_two.shape[1], inputs_two.shape[2]) )
        
        for depth in range(output.shape[0]):
           for a in range(0, self.output_shape[1], self.shape[0] ):
              for b in range(0, self.output_shape[2], self.shape[1] ):
                
                    val = input[depth, a // self.shape[0], b // self.shape[1]] 
                    subSection = inputs_two[depth, a : a + self.shape[0] , b : b + self.shape[1]]
                    max = np.max(subSection)
                    
                    max_index = -1
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                            if max == inputs_two[depth, a + i, b + j]:
                                max_index = (i,j)
                                break
                        else:
                            continue
                        break
                    
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                            if max_index == (i,j):
                                output[depth, a + i, b + j] = val
                            else:
                                output[depth, a + i, b + j] = 0
        return output
    
    def set_input_size(self, layer : Layer):
          self.input_shape = layer.output_shape
          #The 3rd dimension from the last layer is maintained
          self.size = layer.output_shape[0]
          self.output_shape = (layer.output_shape[0], int( (self.input_shape[1] - self.shape[0]) / self.stride + 1 ), int( (self.input_shape[2] - self.shape[1]) / self.stride + 1) )
          
    def save_str(self) -> str:
        string = ""
        string += "LayerType: MaxPool\n"
        string += f"shape: {self.shape}\n"
        string += f"input shape: {self.input_shape}\n"
        string += f"output shape: {self.output_shape}\n"
        string += f"stride: {self.stride}\n"
        
        return string
    
    def load(self, str): 
        #first line is layer name
        lines = str.split("\n")
        self.shape = parse_tuple(lines[1][len("shape: "):])
        self.input_shape = parse_tuple(lines[2][len("input shape: "):])
        self.output_shape = parse_tuple(lines[3][len("output shape: "):])
        self.stride = int(lines[4][len("stride: "):])

class AvgPoolLayer(Layer):
    def __init__(self, shape = None, stride : int = None) -> None:
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
    
    def back_process(self, input, inputs_two):
        #Expanding input to its original size before it was shrinked by the pooling
        output = np.zeros( (input.shape[0], input.shape[1] * self.shape[0], input.shape[2] * self.shape[1]) )
        
        for depth in range(output.shape[0]):
           for a in range(0, self.output_shape[1], self.shape[0] ):
              for b in range(0, self.output_shape[2], self.shape[1] ):
                
                    val = input[depth, a // self.shape[0], b // self.shape[1]] / (self.shape[0] * self.shape[1])
                    
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                                output[depth, a + i, b + j] = val
        return output
    
    def set_input_size(self, layer : Layer):
          self.inputShape = layer.output_shape
          #Size represents how many images we're pooling which corresponds to the 3D dimension of the input image
          self.size = layer.output_shape[0]
          self.output_shape = (self.size, int( (self.inputShape[1] - self.shape[0]) / self.stride + 1 ), int( (self.inputShape[2] - self.shape[1]) / self.stride + 1) )
          
    def save_str(self) -> str:
        string = ""
        string += "LayerType: AvgPool\n"
        string += f"shape: {self.shape}\n"
        string += f"input shape: {self.input_shape}\n"
        string += f"output shape: {self.output_shape}\n"
        string += f"stride: {self.stride}\n"
        
        return string
    
    def load(self, str): 
        #first line is layer name
        lines = str.split("\n")
        self.shape = parse_tuple(lines[1][len("shape: "):])
        self.input_shape = parse_tuple(lines[2][len("input shape: "):])
        self.output_shape = parse_tuple(lines[3][len("output shape: "):])
        self.stride = int(lines[4][len("stride: "):])

class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__(None, None)
       
    def process(self, inputs):
        return np.reshape(inputs, (-1,1))
     
    def back_process(self, inputs, inputs_two):
        #Unflattens the array
        return np.reshape(inputs, self.input_shape)
    
     
    def set_input_size(self, layer : Layer):
        self.input_shape = layer.output_shape
        self.output_shape = ( self.input_shape[0] * self.input_shape[1] * self.input_shape[2], )
        self.size = self.output_shape[0]
        
    def save_str(self) -> str:
        string = ""
        string += "LayerType: Flatten\n"
        string += f"input shape: {self.input_shape}\n"
        string += f"output shape: {self.output_shape}\n"
        string += f"size: {self.size}\n"
        
        return string
    
    def load(self, str): 
        #first line is layer name
        lines = str.split("\n")
        self.input_shape = parse_tuple(lines[1][len("input shape: "):])
        self.output_shape = parse_tuple(lines[2][len("output shape: "):])
        self.size = int(lines[3][len("size: "):])
  