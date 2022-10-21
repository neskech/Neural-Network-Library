from .Layer import Layer, ACT_FUNC
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
        