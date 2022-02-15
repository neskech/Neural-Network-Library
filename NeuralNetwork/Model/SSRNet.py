from NeuralNetwork.Model.NeuralNet import NeuralNet
from NeuralNetwork.Model.NeuralNet import ACT_FUNC
import numpy as np

class SSRNet(NeuralNet):
     
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
        self.cost_function = self.SSR
 #   def __init__(self, fileName):
        #self.load(file_name=fileName)
       
    def SSR(self) -> float:
        sum = 0
        for i in range( len(self.trainX) ):
            predicted = self.evaluate(self.trainX[i])
            for j in range( predicted.shape[1] ):
                sum += ( self.trainY[i] - predicted ) ** 2
        return sum

    def SSR_DERIV(self, data_index, outputs):
        matrix = np.zeros( (len(outputs), 1), dtype=np.float64 )
        #print(f'OUTPUT SHAPEEE {outputs.shape} AND LENGTH {len(outputs)}')
        for a in range( len(outputs) ):
            #print(f'AAAA {a} AND STUFF {-2 * ( self.trainY[data_index] - outputs[a,0] )} AND INDEX OF MATRIX IN A {matrix[a]}')
            matrix[a,0] = -2 * ( self.trainY[data_index] - outputs[a,0] )
        return matrix
            
    def optomize(self):
        
        avgD_weights = [ np.zeros( (a,b), dtype=np.float64 ) for a,b in zip( self.dimensions[1:], self.dimensions[:-1] ) ]
        avgD_biases = [ np.zeros( (a,1), dtype=np.float64 ) for a in self.dimensions[1:] ]

        
        rand_data_points = np.random.randint(0, len(self.trainX), size=self.batch_size)
        for i in rand_data_points:
           inputs = self.trainX[i]
           outputs, acts = self.forward_propagation(inputs)
           cost_deriv = self.SSR_DERIV(i, acts[-1] )
           #print('COST DERIV SHAPE BEFORREEEE',cost_deriv.shape)
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
          
        
            
    
