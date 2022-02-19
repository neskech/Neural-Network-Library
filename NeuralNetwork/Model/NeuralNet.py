
from audioop import bias
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
<<<<<<< HEAD
from NeuralNetwork.Model.Functiions import Classification_accuracy, Regression_accuracy
from NeuralNetwork.Model.Optomizers import Adam_Optomizer, Default_Optomizer, Optomizer
=======
>>>>>>> 4b23bcd1 (Working ANN With New Model)

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
        
<<<<<<< HEAD

    
    
    def compile(self, optomizer, cost, accuracy, debug = False ):
        self.debug = debug
        self.accuracy = Classification_accuracy if accuracy else Regression_accuracy

        match cost:
            case Cost.SQUARE_RESIDUALS:
                self.cost_function = SSR
                self.cost_function_derivative = SSR_Derivative             
=======
        
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
                
>>>>>>> 4b23bcd1 (Working ANN With New Model)
            case Cost.CROSS_ENTROPY:
                self.cost_function = Cross_Entropy
                self.cost_function_derivative = Cross_Entropy_Derivative
                
        match optomizer:
            case Optomizer.DEFAULT:
                self.optomizer = self.Default_Optomizer
            case Optomizer.ADAM:
<<<<<<< HEAD
                self.optomizer = Adam_Optomizer                             
                dimensions = [ a.size for a in self.layers if type(a).__class__ is DenseLayer or type(a).__class__ is FlattenLayer ]
=======
                self.optomizer = self.Adam_Optomizer
                dimensions = [ a.size for a in self.layers if type(a) is DenseLayer or type(a) is FlattenLayer ]
>>>>>>> 4b23bcd1 (Working ANN With New Model)
                weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
                self.prev_momentum_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_momentum_Bias = [ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
                self.prev_EXPWA_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_EXPWA_Bias =[ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]

<<<<<<< HEAD
                if any( [type(a).__class__ is ConvolutionLayer for a in self.layers] ):
                    dims = [ (a.size) + a.kernel_shape for a in self.layers if type(a).__class__ is ConvolutionLayer ]
                    self.prev_momentum_kernel = [ np.zeros(a) for a in dims ]
                    self.prev_EXPA_kernel = [ np.zeros(a) for a in dims ]

    
=======
                if any( [type(a) is ConvolutionLayer for a in self.layers] ):
                    dims = [ a.kernel_shape for a in self.layers if type(a) is ConvolutionLayer ]
                    self.prev_momentum_kernel = [ np.zeros(a) for a in dims ]
                    self.prev_EXPWA_kernel = [ np.zeros(a) for a in dims ]
                    
>>>>>>> 4b23bcd1 (Working ANN With New Model)
    def add(self, layer : Layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[ len(self.layers) - 1].set_input_size( self.layers[ len(self.layers) - 2 ] )
        

    
    def random_restarts(self, numIterations, numRestarts, mean = 0, SD = 1):
            costsAndSeeds = []
            for i in range( numRestarts ):
                seed = random.randint(0,5000)
                self.init_paramaters(mean , SD, False, seed)
                for j in range( numIterations ):
                    self.optomize()
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
            seed = self.random_restarts(numRestart_Iterations)
        self.init_paramaters(0, 1, seed) 
        
        LR_pat = 0
        DB_pat = 0
        prev_cost = 0
        for a in range(numIterations):
            if self.useLearningParams: 
                     prev_cost = self.cost_function()
                
            gradientmag = self.optomizer(X, Y)
                
            if self.useLearningParams:
                  curr_cost = self.cost_function()
                  if curr_cost > prev_cost: 
                      LR_pat += 1
            DB_pat += 1

            if self.debug and DB_pat > self.debug_patience: 
                ##print(f' Cost :: {prev_cost} \nLearning Rate :: {self.learningRate} \n Gradient Mag :: {gradientmag}' )
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
            input = input[np.newaxis, ...]
        acts.append(input)
            
        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
        for index, layer in enumerate(self.layers[start:]):
            typeL = type(layer)
            if typeL is DenseLayer or typeL is ConvolutionLayer:
               outputs.append(layer.process(acts[index]))
               acts.append(layer.activate(outputs[index], use_derivative=False))                
               
            elif typeL is FlattenLayer:
                acts.append(layer.process(outputs[index-1]))
            else:
                outputs.append(layer.process(outputs[index-1]))

        return outputs, acts            
    
<<<<<<< HEAD
    def backwards_propagation(self, cost_deriv, outputs, acts):
        dWeights = []
        dBiases = []
        dKernel = []
        indices = [ i for i in range(len(self.layers)) if type(self.layers[i]).__class__ is ConvolutionLayer or type(self.layers[i]).__class__ is DenseLayer ]
        #Make derib indexes array for each layer with parsaters
        for deriv_index in indices:
            values = cost_deriv
            
            #Make numLayers all the actual layers
            for k in range( self.num_layers - 2, deriv_index - 1, -1 ):  
               #For the first iteration, check if we have activation function on final layer
               #print(f'IN LOOP :: k -> {k} outputs[k] shape -> {outputs[k].shape} weights shape -> {self.weights[k].shape} values shape -> {values.shape}')
               if self.use_last_activation or k < self.num_layers - 2:
                    act_deriv = np.zeros( outputs[k].shape )
                    for a in range( len(act_deriv) ):
                         act_deriv[a][0] = self.activation_deriv( outputs[k][a,0] )     
                         
                    values = np.multiply(values, act_deriv)
                    
               if k != deriv_index:
                   values = np.matmul( self.weights[k].T, values )
                  
            dBiases.append( values ) 
            #print(f'VALUES SHAPE {values.shape} AND ACTS SHAPE {acts[deriv_index].shape} AND DERIV INDEX {deriv_index}')
            dWeights.append( np.matmul(values, acts[deriv_index].T ) )
    
=======
    def backwards_propagation(self, outputs, acts, cost_deriv):
        dbiases = []
        dweights = []
        dkernels = []
        #print('START ', acts[1])
        
        deriv_indices = [i for i in range(len(self.layers[:-1])) if type(self.layers[i]) is ConvolutionLayer or type(self.layers[i]) is DenseLayer]
        first_dense_layer_index = -1
        for index, layer in enumerate(self.layers):
            if type(layer) is DenseLayer:
                first_dense_layer_index = index
                break
         
        ##print('DERIV INDEX ', deriv_indices, 'FIRST ',first_dense_layer_index)   
        for deriv_index in deriv_indices:
            values = cost_deriv
            ##print('DERIVVVV ',deriv_index)
            for k in range(self.num_layers - 1, deriv_index , -1):
                typeL = type(self.layers[k])
                if typeL is DenseLayer or typeL is ConvolutionLayer:
                    act_deriv = self.layers[k].activate(outputs[k-1],use_derivative=True)
                    ##print('ACT DERIV ',act_deriv)
                    if not act_deriv is None:
                         values = np.multiply(act_deriv, values)
                
                if k != deriv_index + 1:
                    values = self.layers[k].back_process(values)
                    ##print('RESULT ',values)
                    
            #If it its a weight we're finding the deriv of
            if deriv_index >= first_dense_layer_index:
                ##print('YEYEYE', [a.shape for a in dweights], 'FINAL ',values, 'ACTS ',acts[deriv_index].T.shape)
                #print('HHEHEHE',deriv_index, ' ', values, ' ACTS ', acts[1] )
                dbiases.append(values)
                dweights.append(np.matmul(values, acts[deriv_index].T))
            #Else its a convolution layer
            else:
                ##print('NONO')
                bias, kernel = self.layers[deriv_index].derive(values)
                dbiases.append(bias)
                dkernels.append(kernel)
            
        return dweights, dbiases, dkernels    
>>>>>>> 4b23bcd1 (Working ANN With New Model)
    
    def freeMemory(self):
         for layer in self.layers:
             if type(layer).__class__ is ConvolutionLayer:
                 del layer.kernel
                 del layer.biases
             elif type(layer).__class__ is DenseLayer:
                 del layer.weights
                 del layer.biases
    
    

    
    
    def evaluate(self, inputs, argMax : bool = False):
        values = None
        if isinstance(inputs, list):
            values = np.array(inputs, dtype=np.float64).reshape( ( len(inputs), 1) )
        else:
           values = inputs.reshape( (inputs.size, 1) )

        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
            
        for index, layer in enumerate(self.layers[start:]):
            ##print(values, index, layer.weights.shape)
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
            indexO = np.where( output == max(output) )
            count += 1 if  indexO == output_index else 0

        return count / len(self.trainY)
     

    def Regression_accuracy(self, testX, testY):      
        sum = 0
        for input_set, output_set in zip(testX, testY):
             output = self.evaluate(input_set)[0]
             sum += (output - output_set) ** 2

        return np.sqrt(sum) / len(self.trainY)
    






    def Adam_Optomizer(self, X, Y):
              
        
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.size,1), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
        print([a.shape for a in avgD_weights])
        print([a.shape for a in avgD_biases])
        
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           ##print('COST DERIV',cost_deriv)
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, cost_deriv)
           ##print([a.shape for a in dweights])
           ##print([a.shape for a in avgD_weights])
           ##print('BIASES ', [a.shape for a in dbiases])
           
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
        print('WEIGHT INDICES ',weight_indices)
        index = 0
        for a in range( len(avgD_weights) ):
            
            EXPWA_Weight = self.EXPWA * self.prev_EXPWA_Weight[a] + (1 - self.EXPWA) * np.square(avgD_weights[a])
            lr_matrix = np.empty( self.layers[ weight_indices[index] ].weights.shape )
            lr_matrix.fill(self.learningRate)
            lr_matrix =  np.divide(lr_matrix, np.sqrt( (EXPWA_Weight + self.epsillon) ) )
            
            changeW =  self.momentum * self.prev_momentum_Weight[a] + (1 - self.momentum) * avgD_weights[a]
            self.layers[ weight_indices[index] ].weights -= np.multiply(changeW, lr_matrix)
            print('change @w', changeW)
            print('avg d @w ',avgD_weights)
            index += 1
            self.prev_momentum_Weight[a] = changeW
            self.prev_EXPWA_Weight[a] = EXPWA_Weight
            
            if self.debug: mag += np.sum( np.square(changeW) )
          
        bias_indices = [i for i in range(len(self.layers)) if (type(self.layers[i]) is DenseLayer and i > 0) or type(self.layers[i]) is ConvolutionLayer]
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
            
            changeK =  self.momentum * self.prev_momentum_kernel[b] + (1 - self.momentum) * avgD_kernels[c] 
            self.layers[ kernel_indices[index] ].kernels -= np.multiply(changeK, lr_matrix)
            index += 1
            self.prev_momentum_kernel[c] = changeK
            self.prev_EXPWA_kernel[c] = EXPWA_Kernel
             
            if self.debug: mag += np.sum( np.square(changeK) )
            
        if self.debug: print(f'GRADIENT MAG :: {np.sqrt(mag)}')





    def Default_Optomizer(self, X, Y): 
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.size,1), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
        print([a.shape for a in avgD_weights])
        print([a.shape for a in avgD_biases])
        
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, cost_deriv)
           
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
        print('WEIGHT INDICES ',weight_indices)
        index = 0
        for a in range( len(avgD_weights) ):
            
            changeW = avgD_weights[a] * self.learningRate
            self.layers[ weight_indices[index] ].weights -= changeW
            print(changeW)
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





  

          
                     


      
    
   