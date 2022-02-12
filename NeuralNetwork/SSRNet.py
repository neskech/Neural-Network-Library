from NeuralNetwork.NeuralNet import NeuralNet
from NeuralNetwork.NeuralNet import ACT_FUNC
from NeuralNetwork.NeuralNet import running_product
import numpy as np

class SSRNet(NeuralNet):
     
    def __init__(self, dimensions : list[int], learning_rate : float, activation_function : ACT_FUNC, debug : bool = False, batch_size : int = 3, momentum : float = 0  ) -> None:
        super().__init__(dimensions, learning_rate, activation_function, debug)
        self.momentum = momentum

        weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
        self.prev_gradient_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.prev_gradient_Bias = [ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
        self.batch_size = batch_size
        
    def __init__(self, fileName):
        self.load(file_name=fileName)
       
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
            changeW = avgD_weights[a] * self.learn_rate + self.momentum * self.prev_gradient_Weight[a]
            self.weights[a] -= changeW
            self.prev_gradient_Weight[a] = changeW
            mag += np.sum( np.multiply(changeW,changeW) )
            
            changeB = avgD_biases[a] * self.learn_rate + self.momentum * self.prev_gradient_Bias[a]
            self.biases[a] -= changeB
            self.prev_gradient_Bias[a] = changeB
            mag += np.sum( np.multiply(changeB,changeB) )
        if self.debug: print(f'GRADIENT MAG :: {np.sqrt(mag)}')
          
        
            
    
