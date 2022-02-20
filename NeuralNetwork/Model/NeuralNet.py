
from audioop import bias
from cmath import cos
from re import U
from matplotlib import use
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as py
from abc import abstractmethod
from enum import Enum
import random

from NeuralNetwork.Layer.Layer import AvgPoolLayer, ConvolutionLayer, DenseLayer, FlattenLayer, Layer, MaxPoolLayer
from NeuralNetwork.Model.Cost import SSR, Cost, Cross_Entropy, Cross_Entropy_Derivative, SSR_Derivative


class Optomizer(Enum):
    DEFAULT = 0
    ADAM = 1



def running_product(list):
    sum = 0
    for i in range( len(list) - 1 ):
        sum += list[i] * list[i + 1]
    return sum

class NeuralNet:
    
    def __init__(self):
        self.useLearningParams = False
        self.layers : list[Layer] = []
        
        
    def compile(self, optomizer, cost, accuracy_type,  debug = False, debug_patience = 0):
        self.debug = debug
        self.debug_patience = debug_patience
        self.num_layers = len(self.layers)
        
        if accuracy_type.lower() == 'regressive':
            self.accuracy = self.Regression_accuracy
        elif accuracy_type.lower() == 'classification':
            self.accuracy = self.Classification_accuracy
        else:
            raise Exception(f'ERROR: \'{accuracy_type}\' is not a type of accuracy')
        
        match cost:
            case Cost.SQUARE_RESIDUALS:
                self.cost_function = SSR
                self.cost_function_derivative = SSR_Derivative
                
            case Cost.CROSS_ENTROPY:
                self.cost_function = Cross_Entropy
                self.cost_function_derivative = Cross_Entropy_Derivative
                
        match optomizer:
            case Optomizer.DEFAULT:
                self.optomizer = self.Default_Optomizer
            case Optomizer.ADAM:

                self.optomizer = self.Adam_Optomizer
                dimensions = [ a.size for a in self.layers if type(a) is DenseLayer or type(a) is FlattenLayer ]
                weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
                biasShapes = [ (a.size,1) for index, a in enumerate(self.layers) if type(a) is ConvolutionLayer or (type(a) is DenseLayer and index > 0)]
                self.prev_momentum_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_momentum_Bias = [ np.zeros( a, dtype=np.float64 ) for a in biasShapes ]
                self.prev_EXPWA_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_EXPWA_Bias =[ np.zeros( a,dtype=np.float64 ) for a in biasShapes ]

                if any( [type(a) is ConvolutionLayer for a in self.layers] ):
                    dims = [ a.kernel_shape for a in self.layers if type(a) is ConvolutionLayer ]
                    self.prev_momentum_kernel = [ np.zeros(a) for a in dims ]
                    self.prev_EXPWA_kernel = [ np.zeros(a) for a in dims ]
                    
    def add(self, layer : Layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[ len(self.layers) - 1].set_input_size( self.layers[ len(self.layers) - 2 ] )
        

    
    def random_restarts(self, X, Y, numIterations, numRestarts, mean = 0, SD = 1):
            costsAndSeeds = []
            for i in range( numRestarts ):
                seed = random.randint(0,5000)
                self.init_paramaters(mean , SD, seed)
                for j in range( numIterations ):
                    self.optomizer(X, Y)
                costsAndSeeds.append( (self.cost_function(), seed) )

            costsAndSeeds = sorted(costsAndSeeds, key= lambda x: x[0], reverse=True)
            return costsAndSeeds[0]
    
    def set_learningRate_settings(self, patience, decrease, min):
        self.useLearningParams = True
        self.lr_patience = patience
        self.lr_decrease = decrease
        self.lr_min = min
    
    def set_hyper_params(self, learningRate, momentum, EWA, epsillon, batch_size = 3):
        self.learningRate = learningRate
        self.momentum = momentum
        self.EXPWA = EWA
        self.epsillon = epsillon
        self.batch_size = batch_size
    
    def init_paramaters(self, mean, SD, seed):
        for layer in self.layers:
            layer.init_rand_params(seed, mean, SD)
            
    def fit(self, X, Y, numIterations, numRestarts = 0, numRestart_Iterations = 0):
        if isinstance(X, list):
            X = np.array(X, dtype=np.float64).reshape( (len(X),1) )
        if isinstance(Y, list):
             Y = np.array(Y, dtype=np.float64).reshape( (len(Y),1) )
             
        seed = 0
        if numRestarts != 0:
            seed = self.random_restarts(X, Y, numRestart_Iterations, numRestart_Iterations)
        self.init_paramaters(0, 1, seed) 
        
        LR_pat = 0
        DB_pat = 0
        prev_cost = 0
        for a in range(numIterations):
            if self.useLearningParams: 
                     prev_cost = self.cost_function(X, Y, self.evaluate)
                
            gradientmag = self.optomizer(X, Y)
                
            if self.useLearningParams:
                  curr_cost = self.cost_function(X, Y, self.evaluate)
                  if curr_cost > prev_cost: 
                      LR_pat += 1
            DB_pat += 1

            if self.debug and DB_pat > self.debug_patience: 
                print(f' Cost :: {prev_cost} \nLearning Rate :: {self.learningRate} \n Gradient Mag :: {gradientmag}' )
                DB_pat = 0           
            if self.useLearningParams and LR_pat > self.lr_patience:
                     self.learn_rate = max( self.learningRate * self.lr_decrease, self.lr_min )
                     LR_pat = 0
        
    
    def display(self, X, Y, subdivisions, rangeX):
        py.scatter(X, Y)
        X = [ (a / subdivisions) * rangeX for a in range( subdivisions ) ]
        #Network outputs data as an array
        Y = [ self.evaluate([a])[0] for a in X ]
        py.plot(X, Y)
        py.show()
    
    
    def forward_propagation(self, input):
        outputs = []
        acts = []
        if isinstance(input, list):
          input = np.array(input, dtype=np.float64).reshape((len(input), 1))
        elif len(input.shape) == 1:
            input = input[..., np.newaxis]
        acts.append(input)
            
        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
        for index, layer in enumerate(self.layers[start:]):
            typeL = type(layer)
            if typeL is DenseLayer or typeL is ConvolutionLayer:
               outputs.append(layer.process(acts[-1]))
               acts.append(layer.activate(outputs[-1], use_derivative=False))                
               
            elif typeL is FlattenLayer:
                #Processing of flatten layer is just the inputs of an ANN
                acts.append(layer.process(outputs[index-1]))
            else:
                outputs.append(layer.process(outputs[index-1]))

        return outputs, acts            
    

    def backwards_propagation(self, outputs, acts, cost_deriv, predicted_index):
        dbiases = []
        dweights = []
        dkernels = []
        ##print('START ', acts[1])
        
        deriv_indices = [i for i in range(len(self.layers[:-1])) if type(self.layers[i]) is ConvolutionLayer or type(self.layers[i]) is DenseLayer or type(self.layers[i]) is FlattenLayer]

         
        #print('DERIV INDEX ', deriv_indices, 'FIRST ',first_dense_layer_index)   
        for deriv_index in deriv_indices:
            values = cost_deriv
            ###print('DERIVVVV ',deriv_index)
            for k in range(self.num_layers - 1, deriv_index, -1):
                typeL = type(self.layers[k])
                #print('index ',k,'layer type',typeL.__class__.__name__,'Values shape ',values.shape)
                if typeL is DenseLayer or typeL is ConvolutionLayer:
                    act_deriv = self.layers[k].activate(outputs[k-1], predicted_index, use_derivative=True)
                    #print('act deriv shape',act_deriv.shape)
                    ###print('ACT DERIV ',act_deriv)
                    if not act_deriv is None:
                         values = np.multiply(act_deriv, values)
                
                if k != deriv_index + 1:
                    values = self.layers[k].back_process(values)
                    ###print('RESULT ',values)
                    
            #If it its a weight we're finding the deriv of
            #TODO just do if type( self.layers[derive_index] ) is DenseLayer
            typeL = type(self.layers[deriv_index])
            if typeL is DenseLayer or typeL is FlattenLayer:
                ###print('YEYEYE', [a.shape for a in dweights], 'FINAL ',values, 'ACTS ',acts[deriv_index].T.shape)
                ##print('HHEHEHE',deriv_index, ' ', values, ' ACTS ', acts[1] )
                dbiases.append(values)
                dweights.append(np.matmul(values, acts[deriv_index].T))
            #Else its a convolution layer
            else:
                ###print('NONO')
                kernel, bias = self.layers[deriv_index].derive(values, acts[deriv_index])
                dbiases.append(bias)
                dkernels.append(kernel)
            
        return dweights, dbiases, dkernels    
    
    def freeMemory(self):
         for layer in self.layers:
             if type(layer).__class__ is ConvolutionLayer:
                 del layer.kernel
                 del layer.biases
             elif type(layer).__class__ is DenseLayer:
                 del layer.weights
                 del layer.biases
    
    

    
    
    def evaluate(self, inputs, argMax : bool = False):
        values = inputs
        if isinstance(inputs, list):
            values = np.array(inputs, dtype=np.float64).reshape( ( len(inputs), 1) )
        elif len(inputs.shape) != 3:
           values = inputs.reshape( (inputs.size, 1) )

        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
            
        for index, layer in enumerate(self.layers[start:]):
            ###print(values, index, layer.weights.shape)
            values = layer.process(values)
            values = layer.activate(values, use_derivative=False)

        values.reshape( values.shape[0], 1 )
        
        if argMax:
            max = values[0]
            maxDex = 0
            for index, a in enumerate(values):
                if a > max:
                    max = a
                    maxDex = index
                    
            return maxDex
        return values
              

   
    def Classification_accuracy(self, testX, testY):
        
        count = 0
        for input_set, output_index in zip(testX, testY):
            output = self.evaluate(input_set)
            
            max = 0
            index = 0
            for ind, val in enumerate(output):
                if val > max:
                    max = val
                    index = ind
                    
            count += 1 if  index == output_index else 0

        return count / len(testY)
     

    def Regression_accuracy(self, testX, testY):      
        sum = 0
        for input_set, output_set in zip(testX, testY):
             output = self.evaluate(input_set)[0]
             sum += (output - output_set) ** 2

        return np.sqrt(sum) / len(testY)
    






    def Adam_Optomizer(self, X, Y):
              
        
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.size,1), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
       # #print([a.shape for a in avgD_weights])
       # #print([a.shape for a in avgD_biases])
        
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           ###print('COST DERIV',cost_deriv)
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, predicted_index=Y[i], cost_deriv=cost_deriv)
           ###print([a.shape for a in dweights])
           ###print([a.shape for a in avgD_weights])
           ###print('BIASES ', [a.shape for a in dbiases])
           #print('bias shapes',[a.shape for a in dbiases])
           #print('avg bias shape',[a.shape for a in avgD_biases])
           
           for a in range( len(dweights) ):
               avgD_weights[a] += dweights[a]
           for a in range( len(dbiases) ):
               avgD_biases[a] += dbiases[a]
           for a in range( len(dkernels) ):
               avgD_kernels[a] += dkernels[a]
               
        for a in range( len(dweights) ):
               avgD_weights[a] /= self.batch_size
        for a in range( len(dbiases) ):
               avgD_biases[a]  /= self.batch_size
        for a in range( len(dkernels) ):
               avgD_kernels[a] /= self.batch_size
        
        mag = 0
        add = 0
        if type(self.layers[0]) is DenseLayer: add = 1
        weight_indices = [i+add for i in range(len(self.layers[add:])) if type(self.layers[i]) is DenseLayer]
        #print('WEIGHT INDICES ',weight_indices)
        index = 0
        for a in range( len(avgD_weights) ):
            
            EXPWA_Weight = self.EXPWA * self.prev_EXPWA_Weight[a] + (1 - self.EXPWA) * np.square(avgD_weights[a])
            lr_matrix = np.empty( self.layers[ weight_indices[index] ].weights.shape )
            lr_matrix.fill(self.learningRate)
            lr_matrix =  np.divide(lr_matrix, np.sqrt( (EXPWA_Weight + self.epsillon) ) )
            
            changeW =  self.momentum * self.prev_momentum_Weight[a] + (1 - self.momentum) * avgD_weights[a]
            self.layers[ weight_indices[index] ].weights -= np.multiply(changeW, lr_matrix)
           # #print('change @w', changeW)
           # #print('avg d @w ',avgD_weights)
            index += 1
            self.prev_momentum_Weight[a] = changeW
            self.prev_EXPWA_Weight[a] = EXPWA_Weight
            
            if self.debug: mag += np.sum( np.square(changeW) )
          
        bias_indices = [i for i in range(len(self.layers)) if (type(self.layers[i]) is DenseLayer and i > 0) or type(self.layers[i]) is ConvolutionLayer]
        #print('bias indices',bias_indices)
        index = 0
        for b in range( len(avgD_biases) ):  
            EXPWA_Bias = self.EXPWA * self.prev_EXPWA_Bias[b] + (1 - self.EXPWA) * np.square(avgD_biases[b])
            lr_matrix = np.empty(  self.layers[ bias_indices[index] ].biases.shape )
            lr_matrix.fill(self.learningRate)
            lr_matrix =  np.divide(lr_matrix,  np.sqrt( (EXPWA_Bias + self.epsillon) ) )
            
            changeB =  self.momentum * self.prev_momentum_Bias[b] + (1 - self.momentum) * avgD_biases[b] 
            self.layers[ bias_indices[index] ].biases -= np.multiply(changeB, lr_matrix)
            index += 1
            self.prev_momentum_Bias[b] = changeB
            self.prev_EXPWA_Bias[b] = EXPWA_Bias
             
            if self.debug: mag += np.sum( np.square(changeB) )
         
        kernel_indices = [i for i in range(len(self.layers)) if type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for c in range( len(avgD_kernels) ):
            EXPWA_Kernel = self.EXPWA * self.prev_EXPWA_kernel[c] + (1 - self.EXPWA) * np.square(avgD_kernels[c])
            lr_matrix = np.empty(  self.layers[ kernel_indices[index] ].kernels.shape )
            lr_matrix.fill(self.learningRate)
            lr_matrix =  np.divide(lr_matrix,  np.sqrt( (EXPWA_Kernel + self.epsillon) ) )
            
            changeK =  self.momentum * self.prev_momentum_kernel[c] + (1 - self.momentum) * avgD_kernels[c] 
            self.layers[ kernel_indices[index] ].kernels -= np.multiply(changeK, lr_matrix)
            index += 1
            self.prev_momentum_kernel[c] = changeK
            self.prev_EXPWA_kernel[c] = EXPWA_Kernel
             
            if self.debug: mag += np.sum( np.square(changeK) )
            
        if self.debug: print(f'GRADIENT MAG :: {np.sqrt(mag)}')





    def Default_Optomizer(self, X, Y): 
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.biases.shape), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
       # #print([a.shape for a in avgD_weights])
       # #print([a.shape for a in avgD_biases])
        
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, predicted_index=Y[i], cost_deriv=cost_deriv)

           for a in range( len(dweights) ):
               avgD_weights[a] += dweights[a]
           for a in range( len(dbiases) ):
               avgD_biases[a] += dbiases[a]
           for a in range( len(dkernels) ):
               avgD_kernels[a] += dkernels[a]
               
        for a in range( len(dweights) ):
               avgD_weights[a] /= self.batch_size
        for a in range( len(dbiases) ):
               avgD_biases[a]  /= self.batch_size
        for a in range( len(dkernels) ):
               avgD_kernels[a] /= self.batch_size
        
        mag = 0
        weight_indices = [i+1 for i in range(len(self.layers[1:])) if type(self.layers[i]) is DenseLayer]
       # #print('WEIGHT INDICES ',weight_indices)
        index = 0
        for a in range( len(avgD_weights) ):
            
            changeW = avgD_weights[a] * self.learningRate
            self.layers[ weight_indices[index] ].weights -= changeW
           # #print(changeW)
            index += 1

            if self.debug: mag += np.sum( np.square(changeW) )
          
        bias_indices = [i for i in range(len(self.layers)) if (type(self.layers[i]) is DenseLayer and i > 0) or type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for b in range( len(avgD_biases) ):  
            
            changeB =  avgD_biases[b] * self.learningRate
            self.layers[ bias_indices[index] ].biases -= changeB
            index += 1
             
            if self.debug: mag += np.sum( np.square(changeB) )
         
        kernel_indices = [i for i in range(len(self.layers)) if type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for c in range( len(avgD_kernels) ):
            
            changeK =  avgD_kernels[c] * self.learningRate
            self.layers[ kernel_indices[index] ].kernels -= changeK
            index += 1
            
            if self.debug: mag += np.sum( np.square(changeK) )
            
        if self.debug: print(f'GRADIENT MAG :: {np.sqrt(mag)}')





  

          
                     


      
    
   