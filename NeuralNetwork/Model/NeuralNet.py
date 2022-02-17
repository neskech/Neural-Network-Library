
import numpy as np
import matplotlib.pyplot as py
from abc import abstractmethod
from enum import Enum
import random

from NeuralNetwork.Layer.Layer import AvgPoolLayer, ConvolutionLayer, DenseLayer, FlattenLayer, Layer, MaxPoolLayer
from NeuralNetwork.Model.Cost import SSR, Cost, Cross_Entropy, Cross_Entropy_Derivative, SSR_Derivative
from NeuralNetwork.Model.Optomizers import Adam_Optomizer, Default_Optomizer, Optomizer




def running_product(list):
    sum = 0
    for i in range( len(list) - 1 ):
        sum += list[i] * list[i + 1]
    return sum

class NeuralNet:
    
    def __init__(self):
        self.useLearningParams = False
        self.layers : list[Layer] = []
        

    
    
    def compile(self, optomizer, cost, debug):
        match cost:
            case Cost.SQUARE_RESIDUALS:
                self.cost_function = SSR
                self.cost_function_derivative = SSR_Derivative
                
                dimensions = [ a.size for a in self.layers if type(a).__class__ is DenseLayer or type(a).__class__ is FlattenLayer ]
                weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
                self.prev_momentum_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_momentum_Bias = [ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
                self.prev_EXPWA_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_EXPWA_Bias =[ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
                
            case Cost.CROSS_ENTROPY:
                self.cost_function = Cross_Entropy
                self.cost_function_derivative = Cross_Entropy_Derivative
                
        match optomizer:
            case Optomizer.DEFAULT:
                self.optomizer = Default_Optomizer
            case Optomizer.ADAM:
                self.optomizer = Adam_Optomizer
    
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
            self.init_paramaters(mean, SD, seed = costsAndSeeds[0][1])
    
    def set_learningRate_settings(self, patience, decrease, min):
        self.useLearningParams = True
        self.lr_patience = patience
        self.lr_decrease = decrease
        self.lr_min = min
        pass
    
    def set_hyper_params(self, learningRate, momentum, EAW, epsillon):
        self.learningRate = learningRate
        self.momentum = momentum
        self.EAW = EAW
        self.epsillon= epsillon
    
    def init_paramters(self, mean, SD, seed):
        for layer in self.layers:
            layer.init_rand_params(seed, mean, SD)
            
    def train(X, Y, numIterations, numRestarts = 0, numRestart_Iterations = 0, batch_size = 1):
        pass
    
    def display(X, Y):
        pass
    
    def animate():
        pass
    
    def forward_propagation(self, input):
        outputs = []
        acts = []
        input = np.array(input, dtype=np.float64).reshape(  (len(input), 1))
        acts.append(input)
        
        for index, layer in enumerate(self.layers):
            if not acts[index] is None:
                  values = layer.process( acts[index] )
            else:
                  values = layer.process( outputs[index] )
                  
            outputs.append( values )
            
            if not type(layer).__class__ is MaxPoolLayer and not type(layer).__class__ is AvgPoolLayer:
              acts.append( layer.activate(values) )
            else:
              acts.append( None )

        return outputs, acts            
    
    def backwards_propagation():
        pass
    
    
    def freeMemory(self):
         for layer in self.layers:
             if type(layer).__class__ is ConvolutionLayer:
                 del layer.kernel
                 del layer.biases
             elif type(layer).__class__ is DenseLayer:
                 del layer.weights
                 del layer.biases
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def evaluate(self, inputs ):
        values = None
        if isinstance(inputs, list):
            values = np.array(inputs, dtype=np.float64).reshape( ( len(inputs), 1) )
        else:
           values = inputs.reshape( (inputs.size, 1) )

        for k in range( self.num_layers - 1 ):
              values = np.matmul( self.weights[k] , values ) + self.biases[k]
              
              if k < self.num_layers - 2 or self.use_last_activation:
                  for a in range( len(values) ):
                     values[a] = self.activation( values[a] )

        return values.reshape( values.shape[0], 1 )

    def forward_propagation(self, input : list[float] ):
        #Outputs are the raw output of a node layer before the activation function
        #In other words, the outputs are what's fed into a node layer before it goes through
        #The activation function. They are like inputs of the kth layer
        #Acts are the raw output of the kth layer with the activation function applied if that layer
        #has one
        inputs = []
        acts = []
        input = np.array(input, dtype=np.float64).reshape(  (len(input), 1))
        acts.append(input)
        #TODO Fix this shit
        for k in range( self.num_layers - 1 ):
           # print(f'BEFORE K {k} WEIGHT SHAPE {self.weights[k].shape} and ACTS SHAPE {acts[k].shape} AND BIASES SHAPE {self.biases[k].shape} ')
            values = np.matmul( self.weights[k], acts[k]) + self.biases[k]
           # print(f'AFTER K {k} WEIGHT SHAPE {self.weights[k].shape} and ACTS SHAPE {acts[k].shape} AND BIASES SHAPE {self.biases[k].shape} VALUES SHAPE {values.shape}')
           
            inputs.append( values )
            if self.use_last_activation or k < self.num_layers - 2:
                for a in range( len(values) ):
                     values[a][0] = self.activation( values[a][0] )
                     
           # print( f'VALUES HEHEHEHEHEHEE AFTER MAPPING {values.shape}')
            acts.append( values )

            
        return inputs, acts
    
    def backwards_propagation(self, outputs : list[np.array], acts : list[np.array], cost_deriv : np.array):
        dweights = []
        dbiases = []
        #print(f'COST DERIVATIVE MATRIX {cost_deriv.shape}')
        for deriv_index in range( self.num_layers - 1 ):
            values = cost_deriv
            
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
                  
            dbiases.append( values ) 
            #print(f'VALUES SHAPE {values.shape} AND ACTS SHAPE {acts[deriv_index].shape} AND DERIV INDEX {deriv_index}')
            dweights.append( np.matmul(values, acts[deriv_index].T ) )

            
        return dweights, dbiases
    
 
    def init_paramaters(self, mean : np.float64, standard_deviation : np.float64, init_biases : bool = False, seed = -1):
        if seed != -1:
             np.random.RandomState(seed)

        for i in range( len (self.weights) ):
            for j in range( len(self.weights[i] ) ):
                for k in range( len(self.weights[i][j] ) ):
                    self.weights[i][j][k] = np.random.normal(mean, standard_deviation )

        if not init_biases: return

        for i in range( len (self.biases) ):
            for j in range( len(self.biases[i] ) ):
                for k in range( len(self.biases[i][j] ) ):
                    self.biases[i][j][k] =  np.random.normal(mean, standard_deviation )

    def accuracy_classif(self, testX, testY):
        if self.trainX is None or self.trainY is None:
            raise Exception('ERROR in accuracy function:: training sets not initialized')
        
        count = 0
        for input_set, output_index in zip(testX, testY):
            output = self.evaluate(input_set)
            indexO = np.where( output == max(output) )
            count += 1 if  indexO == output_index else 0

        return count / len(self.trainY)
     
    def accuracy_regress(self, testX, testY):   
        if self.trainX is None or self.trainY is None:
            raise Exception('ERROR in accuracy function:: training sets not initialized')
        
        sum = 0
        for input_set, output_set in zip(testX, testY):
             output = self.evaluate(input_set)[0]
             sum += (output - output_set) ** 2

        return np.sqrt(sum) / len(self.trainY)
    def train(self, trainX, trainY, num_iterations : int):
        self.trainX = trainX
        self.trainY = trainY
        for i in range( num_iterations ):
            if self.debug: print( f' SSR :: {self.SSR()}' )
            self.optomize()
        
    def train(self, num_iterations : int):
        if self.trainX is None or self.trainY is None:
            raise Exception('ERROR in train function:: training sets not initialized')
        for i in range( num_iterations ):
            if self.debug: print(f' SSR :: {self.SSR()}' )
            self.optomize()
      
    def display_data_and_fit(self, rangeX : float, subdivisions : float):
        if self.trainX is None or self.trainY is None:
            raise Exception('ERROR in display function:: training sets not initialized')
        elif self.dimensions[-1] != 1:
            raise Exception('ERROR in display function:: network output must be 1 dimensional')
        
        py.scatter(self.trainX, self.trainY)
        X = [ (a / subdivisions) * rangeX for a in range( subdivisions ) ]
        #Network outputs data as an array
        Y = [ self.evaluate(np.array([a]))[0] for a in X ]
        py.plot(X, Y)
        py.show()
          
                     
    def train_and_animate(self, subdivisions : float, num_iterations : int, rangeX, delay : float = 0.000000005):
        if self.trainX is None or self.trainY is None:
            raise Exception('ERROR in train_and_andimate function:: training sets not initialized')
        
        #rangeX = ( min(self.trainX), max(self.trainX ) )
        xData = []
        yData = []
        
        
        for i in range( num_iterations ):
            py.clf()
            
            self.optomize()
            if self.debug: print(f' SSR :: {self.SSR()}' )
            
            xData.append( [ (a / subdivisions) * (rangeX[1] - rangeX[0]) for a in range( subdivisions ) ] )
            yData.append( [ self.evaluate([a])[0] for a in xData[i] ] )
            py.plot(xData[i],yData[i])
            py.scatter(self.trainX,self.trainY)
            py.pause(delay)
            
            if self.use_lr_Tuning:
                self.learn_rate = max( self.learn_rate * self.lr_decrease, self.lr_min )
            

        py.show()
        
    def train_and_random_restarts(self, num_iterations, num_test_iterations, num_restarts, mean = 0, SD = 1 ):
            if self.trainX is None or self.trainY is None:
                raise Exception('ERROR in train function:: training sets not initialized')

            SSR_and_seed_list = []
            for i in range( num_restarts ):
                seed = random.randint(0,5000)
                self.init_paramaters(mean , SD, False,seed)
                for j in range( num_test_iterations ):
                    self.optomize()
                SSR_and_seed_list.append( (self.cost_function(), seed) )

            SSR_and_seed_list = sorted(SSR_and_seed_list, key= lambda x: x[0], reverse=True )
            self.init_paramaters(mean, SD, seed=SSR_and_seed_list[0][1] )
            
            pat = 0
            for i in range( num_iterations ):
                
                if self.use_lr_Tuning: 
                     prev_cost = self.cost_function()
                     if self.debug: print(f' SSR :: {prev_cost} \nLR :: {self.learn_rate}' )
                
                self.optomize()
                
                if self.use_lr_Tuning:
                  curr_cost = self.cost_function()
                  if curr_cost > prev_cost: 
                      pat += 1

                if self.use_lr_Tuning and pat > self.lr_patience:
                     self.learn_rate = max( self.learn_rate * self.lr_decrease, self.lr_min )
                     pat = 0


    @abstractmethod
    def optomize(self):
        pass
    
    def set_training_data(self, trainX, trainY):
        if isinstance(trainX, list):
             self.trainX = np.array(trainX, dtype=np.float64).reshape( (len(trainX),1) )
        else:
             self.trainX = trainX
             
        if isinstance(trainY, list):
             self.trainY = np.array(trainY, dtype=np.float64).reshape( (len(trainY),1) )
        else:
             self.trainY = trainY
        
    def init_WB(self, weights, biases):
       self.weights = weights
       self.biases = biases
       
    def set_learning_params(self, useTuning : bool, decrease : float, patience : int, min : float):
          self.use_lr_Tuning = useTuning
          self.lr_decrease = decrease
          self.lr_patience = patience
          self.lr_min = min
          
    def save(self, file_name):
        with open(file_name, 'w') as f:
            f.write(f'LAYER_SIZES: {self.dimensions}\n')
            
            if (self.FUNC == ACT_FUNC.RELU):
                f.write('ACT FUNC: RELU\n')
            elif (self.FUNC == ACT_FUNC.SOFT_PLUS):
                f.write('ACT FUNC: SOFT PLUS\n')
            elif (self.FUNC == ACT_FUNC.SIGMOID):
                f.write('ACT FUNC: SIGMOID\n')
            elif (self.FUNC == ACT_FUNC.TANH):
                f.write('ACT FUNC: TANH\n')
                
            f.write(f'USE LAST ACTIVATION: {self.use_last_activation}\n')
            f.write('WEIGHTS:' + '-'*20 + '\n')
            for weight in self.weights:
                for val in np.nditer(weight):
                    f.write(str(val) + '\n')
                    
            f.write('BIASES:' + '-'*20 + '\n')
            for bias in self.biases:
                for val in np.nditer(bias):
                    f.write(str(val) + '\n')
                    
        
    def load(self, file_name):
        lines = open(file_name,'r').readlines()
        length = len('LAYER_SIZES: ')
        dimenSTR = lines[0][ length + 1 : -2 ].split(',')
        self.dimensions = [ int(char.strip()) for char in dimenSTR ]
        self.num_layers = len(self.dimensions)
        
        if lines[1].find('RELU') != -1:
            self.FUNC = ACT_FUNC.RELU
        elif lines[1].find('SOFT PLUS') != -1:
            self.FUNC = ACT_FUNC.SOFT_PLUS
        elif lines[1].find('SIGMOID') != -1:
            self.FUNC = ACT_FUNC.SIGMOID
        elif lines[1].find('TANH') != -1:
            self.FUNC = ACT_FUNC.TANH
        
        self.use_last_activation = bool( lines[2][ len('USE LAST ACTIVATION: ') : ] )
        
        weightShapes = [ (a,b) for a,b in zip(self.dimensions[1:],self.dimensions[:-1]) ]
        weightSizes = [ a * b for a,b in zip(self.dimensions[1:], self.dimensions[:-1]) ]
        biasSizes = [ a for a in self.dimensions[1:] ]
        self.weights = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.biases = [ np.zeros( (a,1),dtype=np.float64 ) for a in self.dimensions[1:] ]

        isWeight = True
        index = 0
        for line in lines[4:]:
            if line.find('BIASES') != -1:
                isWeight = False
                index = 0
                continue
            
            if isWeight:
                weightIndex = 0
                sum = weightSizes[0]
                while index >= sum:
                    weightIndex += 1
                    sum += weightSizes[weightIndex]
                    
                ind = index
                if sum != weightSizes[0]: ind = index -  (sum - weightSizes[weightIndex  ])
                
                row = ind // self.weights[weightIndex].shape[1]
                col = ind % self.weights[weightIndex].shape[1]
                #print(f' ROW {row} COL {col} INDEX {index} IND {ind} WEIGHTINDEX {weightIndex} weights at index {self.weights[weightIndex].shape}')
                self.weights[weightIndex][row,col] = float(line)
                
            else:
                biasIndex = 0
                sum = biasSizes[0]
                while index >= sum:
                    biasIndex += 1
                    sum += biasSizes[biasIndex]
                    
                ind = index
                if sum != biasSizes[0]: ind = index - (sum - biasSizes[biasIndex ])
                
                row = ind 
               # print(f' ROW {row} INDEX {index} IND {ind} WEIGHTINDEX {biasIndex} biases at index {self.biases[biasIndex].shape}')
                self.biases[biasIndex][row,0] = float(line)
  
            index += 1
      
    
   