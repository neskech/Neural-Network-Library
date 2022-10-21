from .Layer import Layer, ACT_FUNC, activation_to_str, str_to_activation, parse_tuple
import numpy as np

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
        return np.matmul(self.weights, inputs) + self.biases
    
    def back_process(self, inputs, inputs_two):
        #Unflattens the array
        return np.matmul(self.weights.T, inputs)
             
    def set_input_size(self, layer : Layer):
        if self.weights is None or self.biases is None:
              self.biases = np.zeros( shape= (self.size, 1), dtype=np.float64 )
              self.weights = np.zeros( shape=(self.size, layer.size), dtype=np.float64 )
              self.output_shape = (self.size)
              self.input_shape = layer.size
              
    def save_str(self) -> str:
        string = ""
        string += "LayerType: Dense\n"
        string += f"size: {self.size}\n"
        string += f"input size: {self.input_shape}\n"
        act_name = activation_to_str(self.activation_func)
        string += f"Activation Function: {act_name}\n"
        
        string += "Weights:\n" 
        string += f"{self.weights}\n"
        string += "Biases:\n" 
        string += f"{self.biases}\n"
        
        return string
        
    def load(self, str):
        lines = str.split("\n")
        #first line is layer name
        
        size_str = lines[1][6:]
        if size_str == "None":
            self.size = None
        else:
            self.size = int(lines[1][6:])
            
        #not a tuple this time
        input_shape_str = lines[2][len("input size: "):]
        if input_shape_str == "None":
            self.input_shape = None
        else:
             self.input_shape = int(lines[2][len("input size: "):])
             
        self.output_shape = (self.size)
        
        activation_str = lines[3][len("Activation Function: "):]  
        self.activation_func = str_to_activation(activation_str) #Handles the None case
        self.setActivation(self.activation_func)
        
        assert(lines[4].find("Weights:") != -1)
        
        a = 5
        
        if lines[a].find("None") != -1:
            self.weights = None
            self.biases = None
            return
        
        self.weights = np.zeros( shape=(self.size, self.input_shape), dtype=np.float64 )
        self.biases = np.zeros( shape=(self.size, 1), dtype=np.float64 )
        
        b = 0
        idx = -1
        while lines[a].find("Biases") == -1:
            if lines[a].find("[") != -1:
                b = 0
                idx += 1
                
            while lines[a].find("[") != -1:
                lines[a] = lines[a][1:]
            while lines[a].find("]") != -1:
                lines[a] = lines[a][:-1]   
                    
            lines[a] = lines[a].strip()
            
            lines[a] = lines[a].split(" ")
            
            while "" in lines[a]:
                idx_ = lines[a].index("")
                lines[a] = lines[a][:idx_] + lines[a][idx_ + 1:]

            for piece in lines[a]:
                c = (b) % self.weights.shape[1]
                self.weights[idx, c] = float(piece)
                b += 1
   
            a += 1
            
        assert(lines[a].find("Biases:") != -1)
        a += 1
        shift = a
        
        while a < len(lines) and len(lines[a].strip()) != 0:
            while lines[a].find("[[") != -1:
                lines[a] = lines[a][1:]
            while lines[a].find("]]") != -1:
                lines[a] = lines[a][:-1]
                
            lines[a] = (lines[a].strip()[1:-1]).strip()
            
            self.biases[a - shift, 0] = float(lines[a])
            a += 1
