
from audioop import bias
import numpy as np
import matplotlib.pyplot as py
from abc import abstractmethod
from enum import Enum
import random

from pint import test





class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2

def running_product(list):
    sum = 0
    for i in range( len(list) - 1 ):
        sum += list[i] * list[i + 1]
    return sum

class NeuralNet:
    
    def __init__(self, dimensions : list[int], learning_rate : np.float64, activation_function : ACT_FUNC, debug : bool = False):
        self.FUNC = activation_function
        self.dimensions = dimensions
        self.num_layers = len(dimensions)
        self.learn_rate = learning_rate

        weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
        self.weights = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.biases = [ np.zeros( (a,1),dtype=np.float64 ) for a in dimensions[1:] ]
        
        self.trainX = None
        self.trainY = None
        self.debug = debug
        self.use_last_activation = True
        np.seterr(all='print')
     
    
    
    def activation(self, x : np.float64):
        if self.FUNC is ACT_FUNC.RELU:
            return max(0,x)

        elif self.FUNC is ACT_FUNC.SOFT_PLUS:
             x = np.clip(x,-100,100)
             return np.log( 1 + np.e ** x )

        elif self.FUNC is ACT_FUNC.SIGMOID:
            x = np.clip(x,-100,100)
            return  1  / ( 1 + np.e ** -x )
    
    def activation_deriv(self, x : np.float64):
        if self.FUNC is ACT_FUNC.RELU:
            return 1 if x >= 0 else 0

        elif self.FUNC is ACT_FUNC.SOFT_PLUS:
                x = np.clip(x,-100,100)
                return (np.e ** x) / ( 1 + np.e ** x )

        elif self.FUNC is ACT_FUNC.SIGMOID:
            x = np.clip(x,-100,100)
            return  ( np.e ** -x ) / ( ( 1 + np.e ** -x) ** 2 ) 

    
    def evaluate(self, inputs : list[np.float64]):
        values = np.array(inputs, dtype=np.float64).reshape( ( len(inputs), 1) )

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
        for input_set, output_set in zip(testX, testY):
            output = self.evaluate(input_set)
            indexO = np.where( output == max(output) )
            output_set = [output_set]
            indexP = np.where( output_set == max(output_set) )
            count += 1 if  indexO == indexP else 0

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
          
                     
    def train_and_animate(self, subdivisions : float, num_iterations : int, rangeX, delay : float = 0.00005):
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
            yData.append( [ self.evaluate([a], use_final_act_func= False)[0] for a in xData[i] ] )
            py.plot(xData[i],yData[i])
            py.scatter(self.trainX,self.trainY)
            py.pause(delay)
            
            if self.use_lr_Tuning:
                self.learn_rate = max( self.learn_rate * self.lr_decrease, self.lr_min )
            

        py.show()
        
    def train_and_random_restarts(self, num_iterations, num_test_iterations, num_restarts, mean = -1, SD = 1 ):
            if self.trainX is None or self.trainY is None:
                raise Exception('ERROR in train function:: training sets not initialized')

            SSR_and_seed_list = []
            for i in range( num_restarts ):
                seed = random.randint(0,5000)
                self.init_paramaters(mean , SD, False,seed)
                for j in range( num_test_iterations ):
                    self.optomize()
                SSR_and_seed_list.append( (self.SSR(), seed) )

            SSR_and_seed_list = sorted(SSR_and_seed_list, key= lambda x: x[0], reverse=True )
            self.init_paramaters(mean, SD, seed=SSR_and_seed_list[0][1] )
            
            pat = 0
            for i in range( num_iterations ):
                prev_cost = self.SSR()
                if self.debug: print(f' SSR :: {prev_cost} \nLR :: {self.learn_rate}' )
                
                self.optomize()
                curr_cost = self.SSR()
                if curr_cost > prev_cost: 
                    pat += 1

                if self.use_lr_Tuning or pat > self.lr_patience:
                     self.learn_rate = max( self.learn_rate * self.lr_decrease, self.lr_min )
                     pat = 0


    @abstractmethod
    def optomize(self):
        pass
    
    def set_training_data(self, trainX, trainY):
        self.trainX = trainX
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
            f.write(f'LAYER_SIZES :: {self.dimensions}\n')
            
            if (self.FUNC == ACT_FUNC.RELU):
                f.write('ACT FUNC :: RELU\n')
            elif (self.FUNC == ACT_FUNC.SOFT_PLUS):
                f.write('ACT FUNC :: SOFT PLUS\n')
            elif (self.FUNC == ACT_FUNC.SIGMOID):
                f.write('ACT FUNC :: SIGMOID\n')
                
            f.write(f'USE LAST ACTIVATION :: {self.use_last_activation}\n')
            f.write('WEIGHTS ::\n')
            for weight in self.weights:
                for val in np.nditer(weight):
                    f.write(str(val) + '\n')
                    
            f.write('BIAS ::\n')
            for bias in self.biases:
                for val in np.nditer(bias):
                    f.write(str(val) + '\n')
                    
        
    def load(self, file_name):
        lines = open(file_name,'r').readlines()
        length = len('LAYER_SIZES :: ')
        dimenSTR = lines[0][ length + 1 : -2 ].split(',')
        self.dimensions = [ int(char.strip()) for char in dimenSTR ]
        self.num_layers = len(self.dimensions)
        
        if lines[1].find('RELU') != -1:
            self.FUNC = ACT_FUNC.RELU
        elif lines[1].find('SOFT PLUS') != -1:
            self.FUNC = ACT_FUNC.SOFT_PLUS
        elif lines[1].find('SIGMOID') != -1:
            self.FUNC = ACT_FUNC.SIGMOID
        
        self.use_last_activation = bool( lines[2][ len('USE LAST ACTIVATION :: ') : ] )
        
        weightShapes = [ (a,b) for a,b in zip(self.dimensions[1:],self.dimensions[:-1]) ]
        weightSizes = [ a * b for a,b in zip(self.dimensions[1:], self.dimensions[:-1]) ]
        biasSizes = [ a for a in self.dimensions[1:] ]
        self.weights = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
        self.biases = [ np.zeros( (a,1),dtype=np.float64 ) for a in self.dimensions[1:] ]

        isWeight = True
        index = 0
        for line in lines[4:]:
            if line.find('BIAS') != -1:
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
      
    
   