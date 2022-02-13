from NeuralNetwork.NeuralNet import NeuralNet
from NeuralNetwork.NeuralNet import ACT_FUNC
import numpy as np

class CENet(NeuralNet):
     def __init__(self, dimensions : list[int], learning_rate : float, activation_function : ACT_FUNC, debug : bool = False, batch_size : int = 3, momentum : float = 0, EXPWA : float = 0, epsillon : float = 0.00001  ) -> None:
        super().__init__(dimensions, learning_rate, activation_function, debug)
        self.momentum = momentum

        weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
        self.prev_momentum_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.prev_momentum_Bias = [ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
        self.prev_EXPWA_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.prev_EXPWA_Bias =[ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
        
        self.batch_size = batch_size
        self.EXPWA = EXPWA
        self.episllon = epsillon
        self.cost_function = self.cross_entropy_cost
    # def __init__(self, fileName):
     #    self.load(file_name=fileName)
        
        
     def argMax(self, values, threshold : float = 0.5):
         for a in range( values.shape[0] ):
             values[a,0] = 1 if values[a,0] >= threshold else 0
         return values

     def softMax(self, values):
         for a in range( values.shape[0] ):
             numerator = np.e ** values[a,0]
             denominator = 0
             
             for b in range( values.shape[0] ):
                 denominator += np.e ** values[b,0]
             values[a,0] = numerator / denominator
           
             
     def cross_entropy_deriv(self, output_neuron_number, output_values):
         #Output neuron number is the observed value of the data point
         matrix = np.zeros( (self.dimensions[-1], 1) )
         for a in range( self.dimensions[-1]  ):
             if a == output_neuron_number:
                 matrix[a,0] = output_values[a,0] - 1
             else:
                 matrix[a,0] = output_values[a,0]
         return matrix
     
     def cross_entropy(self, predicted_value):
         return -np.log(predicted_value)
     
     def cross_entropy_cost(self):
           sum = 0
           rand_data_points = np.random.randint(0, len(self.trainX), size=self.batch_size)
           for i in rand_data_points:
               outputs = self.soft_evaluate(self.trainX[i])
               output_index = int(self.trainY[i])
               sum += self.cross_entropy(outputs[output_index])
           return sum / len(rand_data_points)
               
         
     def evaluate(self, inputs):
         stuff = super().evaluate(inputs)
         self.softMax(stuff)
       #  print(f'STUFF {stuff}')
         max = stuff[0,0]
         index = 0
         for a in range( stuff.shape[0] ):
             if stuff[a,0] > max:
                 max = stuff[a,0]
                 index = a
             
         return index
     
     def soft_evaluate(self, inputs):
         stuff = super().evaluate(inputs)
         self.softMax(stuff)
         return stuff
     
     def accuracy(self, testX, testY):
         if self.trainX is None or self.trainY is None:
              raise Exception('ERROR in train function:: training sets not initialized')
          
         count = 0
         for input_set, output_index in zip(testX, testY):
            outputs = self.soft_evaluate(input_set)
            
            index = 0
            max = outputs[0,0]
            for a in range( outputs.shape[0] ):
               if outputs[a,0] > max:
                 max = outputs[a,0]
                 index = a
                 
            count += 1 if index == output_index else 0
         return count / len( testY )
         
            
     def optomize(self):
        
        avgD_weights = [ np.zeros( (a,b), dtype=np.float64 ) for a,b in zip( self.dimensions[1:], self.dimensions[:-1] ) ]
        avgD_biases = [ np.zeros( (a,1), dtype=np.float64 ) for a in self.dimensions[1:] ]

        
        rand_data_points = np.random.randint(0, len(self.trainX), size=self.batch_size)
        for i in rand_data_points:
           inputs = self.trainX[i]
           outputs, acts = self.forward_propagation(inputs)
           self.softMax(acts[-1])
           output_index = self.trainY[i]
           
           cost_deriv = self.cross_entropy_deriv( output_neuron_number=output_index, output_values=acts[-1] )
           dweights, dbiases = self.backwards_propagation(outputs, acts, cost_deriv)
           
           for a in range( len(dweights) ):
               avgD_weights[a] += dweights[a]
               avgD_biases[a] += dbiases[a]
               
        for a in range( len(dweights) ):
            avgD_weights[a] /= self.batch_size
            avgD_biases[a] /=  self.batch_size
        
        mag = 0
        for a in range( len(self.weights) ):
            
            EXPWA_Weight = self.EXPWA * self.prev_EXPWA_Weight[a] + (1 - self.EXPWA) * np.square(avgD_weights[a])
            lr_matrix = np.empty( self.weights[a].shape )
            lr_matrix.fill(self.learn_rate)
            lr_matrix =  np.divide(lr_matrix,  np.sqrt( (EXPWA_Weight + self.episllon) ) )
            
            changeW =  self.momentum * self.prev_momentum_Weight[a] + (1 - self.momentum) * avgD_weights[a]
            self.weights[a] -= np.multiply(changeW, lr_matrix)
            self.prev_momentum_Weight[a] = changeW
            self.prev_EXPWA_Weight[a] = EXPWA_Weight
            
            if self.debug: mag += np.sum( np.square(changeW) )
            
            EXPWA_Bias = self.EXPWA* self.prev_EXPWA_Bias[a] + (1 - self.EXPWA) * np.square(avgD_biases[a])
            lr_matrix = np.empty( self.biases[a].shape )
            lr_matrix.fill(self.learn_rate)
            lr_matrix =  np.divide(lr_matrix,  np.sqrt( (EXPWA_Bias + self.episllon) ) )
            
            changeB =  self.momentum * self.prev_momentum_Bias[a] + (1 - self.momentum) * avgD_biases[a] 
            self.biases[a] -= np.multiply(changeB, lr_matrix)
            self.prev_momentum_Bias[a] = changeB
            self.prev_EXPWA_Bias[a] = EXPWA_Bias
             
            if self.debug: mag += np.sum( np.square(changeB) )
            
        if self.debug: print(f'GRADIENT MAG :: {np.sqrt(mag)}')
          
        