import numpy as np
import matplotlib.pyplot as py
from enum import Enum
from time import time
import random
from .Activations import argMax
from NeuralNetwork.Layer.Layer import Layer
from NeuralNetwork.Layer.Misc import AvgPoolLayer, MaxPoolLayer, FlattenLayer
from NeuralNetwork.Layer.Conv import ConvolutionLayer
from NeuralNetwork.Layer.Dense import DenseLayer
from NeuralNetwork.Model.Cost import Square_Residuals, Cost, Cross_Entropy, Cross_Entropy_Derivative, Square_Residuals_Derivative


class Optomizer(Enum):
    DEFAULT = 0
    ADAM = 1
    
class AccuracyTP(Enum):
    REGRESSION = 0
    CLASSIFICATION = 1

class Model:
    
    def __init__(self):
        #Learning Rate Tuning / Scheduling
        self.useLearningRateTuning = False
        #layers
        self.layers : list[Layer] = []
        
        #Data to be graphed after training
        self.loss_metrics = []
        self.gradient_mag_metrics = []
       
    def add(self, layer : Layer):
        """Add a layer to the network

        Args:
            layer (Layer): A layer of The User's Choice
        """
        
        self.layers.append(layer)
        if len(self.layers) > 1:
            #configure the input size of the layer we just added
            #by referring to the previous later
            self.layers[len(self.layers) - 1].set_input_size(self.layers[len(self.layers) - 2])
        
    def set_learningRate_tuning(self, patience: int, decrease: float, min: float):
        """Set Learning Rate Tuning Paramaters For This Model. Turned off By Default

        Args:
            patience (_type_): How Long to Wait Until Learning Rate is Decremented
            decrease (_type_): The Proportion to Multiply the Learning Rate by Each Decrement
            min (_type_): The Minimum Allowable Value of the Learning Rate After Many Decrements
        """
        self.useLearningParams = True
        self.lr_patience = patience
        self.lr_decrease = decrease
        self.lr_min = min
        
    def set_hyper_params(self, learning_rate: float, batch_size: int, momentum: float = 0.9, momentum2: float = 0.999, epsillon: float = 1e-8):
        """Set the Hyper Paramaters of the Model. Adam Paramaters Only Used if Adam is Enabled in the Compile() Function

        Args:
            learningRate (_type_): How Fast the Model Learns
            batch_size (int, optional): _description_. Defaults to 3.
            momentum (_type_): Momentum Term For Adam Optimization
            momentum2 (_type_): Second Momentum Term For Adam Optimization
            epsillon (_type_): Epsillon Term For Adam Optimization
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.momentum2 = momentum2
        self.epsillon = epsillon
        
    def compile(self, optomizer: Optomizer, loss_function: Cost, accuracy_type: AccuracyTP,  debug: bool = False, debug_patience: int = 0):
        """ Sets core data for a model such as the loss function, optomizer, and accuracy type. 
        Call after adding all layers and setting hyper-paramaters

        Args:
            optomizer (Optomizer): Adam or Default
            loss_function (Cost): The Loss Function
            accuracy_type (str): Either "regressive" or "classification"
            debug (bool, optional): Printing of Debug Information During Training. Defaults to False.
            debug_patience (int, optional): Delay (In Iterations) Between Printing of Debug Information. Defaults to 0.
        """
        #bool for printing of debug information
        self.debug = debug
        #delay between iterations to print debug information
        self.debug_patience = debug_patience
        
        #number of layers
        self.num_layers = len(self.layers)
        
        #set function pointers for accuracy
        match accuracy_type:
            case AccuracyTP.REGRESSION:
                self.accuracy = self.loss_on_dataset
            case AccuracyTP.CLASSIFICATION:
                self.accuracy = self.classification_accuracy
            case _:
                raise Exception(f'ERROR: \'{accuracy_type}\' is not supported')
        
        #set function pointers for loss function
        match loss_function:
            case Cost.SQUARE_RESIDUALS:
                self.cost_function = Square_Residuals
                self.cost_function_derivative = Square_Residuals_Derivative
                
            case Cost.CROSS_ENTROPY:
                self.cost_function = Cross_Entropy
                self.cost_function_derivative = Cross_Entropy_Derivative
                
            case _:
                raise Exception(f'ERROR: \'{loss_function}\' is not supported')
                
        
        #setup the member variables needed for the optomizers 
        match optomizer:
            case Optomizer.DEFAULT:
                self.optomizer = self.Default_Optomizer
                
            case Optomizer.ADAM:

                self.optomizer = self.Adam_Optomizer
                
                #Build up the dimensions for our matrices
                dimensions = [ a.size for a in self.layers if type(a) is DenseLayer or type(a) is FlattenLayer ]
                weightShapes = [ (a,b) for a,b in zip(dimensions[1:], dimensions[:-1]) ]
                biasShapes = [ (a.size,1) for index, a in enumerate(self.layers) if type(a) is ConvolutionLayer or (type(a) is DenseLayer and index > 0)]
                
                #Momentum
                self.prev_momentum_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_momentum_Bias = [ np.zeros( a, dtype=np.float64 ) for a in biasShapes ]
                #EXPWA = Exponentially Weighted Average
                self.prev_EXPWA_Weight = [ np.zeros(a,dtype=np.float64) for a in weightShapes ]
                self.prev_EXPWA_Bias = [ np.zeros( a,dtype=np.float64 ) for a in biasShapes ]

                #For any convolutional layers, set up their matrices as well
                if any( [type(a) is ConvolutionLayer for a in self.layers] ):
                    dims = [ a.kernel_shape for a in self.layers if type(a) is ConvolutionLayer ]
                    self.prev_momentum_kernel = [ np.zeros(a) for a in dims ]
                    self.prev_EXPWA_kernel = [ np.zeros(a) for a in dims ]
                
    
    def init_paramaters(self, mean: float, SD: float, seed: float):
        """Initiates the Paramaters of the Model

        Args:
            mean (float): Mean For Sampling Values
            SD (float): Standard Deviation For Sampling Values
            seed (float): Seed For Sampling Values (Can Get A Desirable Seed from the Random_Restarts() Function)
        """
        for layer in self.layers:
            layer.init_rand_params(seed, mean, SD)
            
    def random_restarts(self, trainX, trainY, num_restarts: int, epochs: int, mean: float = 0, SD: float = 1) -> float:
            """Starts Short Training Sequences With Several Different Seeds, Then Returns the most Desirable Seed
            of the Generated List

            Args:
                trainX (_type_): Training Inputs
                trainY (_type_): Training Outputs
                num_restarts (int): Number of Training Sequences to Operate When Making Seed
                epochs (int): Number of Training Epochs per Restart When Making Seed
                mean (float, optional): For Initiating Paramaters. Defaults to 0.
                SD (float, optional): For Initiating Paramaters. Defaults to 1.

            Returns:
                float: The Most Desirable Seed From The Generated List 
            """
            costsAndSeeds = []
            
            for _ in range(num_restarts):
                seed = random.randint(0,5000)
                self.init_paramaters(mean, SD, seed)
                
                for _ in range(epochs):
                    self.optomizer(trainX, trainY)
                    
                costsAndSeeds.append((self.cost_function(), seed))

            costsAndSeeds = sorted(costsAndSeeds, key=lambda x: x[0], reverse=True)
            return costsAndSeeds[0][1]
    
     
    def evaluate(self, inputs, argMax : bool = False):
        """Evaluates the Model on a Given Input

        Args:
            inputs (_type_): Input Matrix to the Model
            argMax (bool, optional): Whether to Argmax the Output. Defaults to False.

        Returns:
            _type_: _description_
        """
        values = inputs
        
        #If the input is a list, convert it to a numpy array
        if isinstance(inputs, list):
            values = np.array(inputs, dtype=np.float64).reshape((len(inputs), 1))
        elif len(inputs.shape) == 1:
           #Add a new axis to the input, making it a column vector and not just a flat list
           values = inputs.reshape((inputs.size, 1))

        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
            
        #Process the inputs through the network
        for index, layer in enumerate(self.layers[start:]):
            values = layer.process(values)
            values = layer.activate(values, use_derivative=False)

        values.reshape(values.shape[0], 1)
        
        #Apply argmax
        if argMax:
            max = values[0]
            maxDex = 0
            for index, a in enumerate(values):
                if a > max:
                    max = a
                    maxDex = index
                    
            return maxDex
        
        return values
           
    def train(self, trainX, trainY, epochs, num_restarts = 0, restart_Epochs = 5):
        """Trains the Model On A Given Dataset

        Args:
            trainX (_type_): Training Set Inputs
            trainY (_type_): Training Set Outputs
            epochs (_type_): Number of Iterations to Train Model For
            numRestarts (int, optional): To be Used in Random_Restarts() Function. Defaults to 0.
            numRestart_Iterations (int, optional): To be Used in Random_Restarts() Function. Defaults to 0..
        """
        
        #If the dataset isn't in the form of a numpy array
        if isinstance(trainX, list):
             trainX = np.array(trainX, dtype=np.float64).reshape( (len(trainX),1) )
        if isinstance(trainY, list):
             trainY = np.array(trainY, dtype=np.float64).reshape( (len(trainY),1) )
            
        #Grab a seed from random restarts 
        seed = random.randint(0,5000)
        if num_restarts != 0:
            seed = self.random_restarts(trainX, trainY, num_restarts, restart_Epochs)
        self.init_paramaters(0, 1, seed) 
        
        lr_pat = 0 #Learning Rate Patience
        db_pat = 0 #Debug Printing Patience
        prev_cost = None #Cost / Loss of Previous Epoch, Used for Learning Rate Tuning
        
        #TODO store loss and gradient mag data in an array for use to graph later
        self.loss_metrics.clear()
        self.gradient_mag_metrics.clear()
        
        cumulative_time = 0
        for e in range(epochs):
            
            start_time = time()
            
            #If Tuning is enabled, want to compare the previous loss to the loss after back propogation
            if self.useLearningRateTuning: 
                     prev_cost = self.cost_function(trainX, trainY, self.evaluate)
            
            #Runs through the dataset using stochastic gradient descent
            #Updates the Paramaters of the model and returns the magnitude of the gradients
            gradientmag = self.optomizer(trainX, trainY)
            self.gradient_mag_metrics.append(gradientmag)
            
            #calculate current cost 
            curr_cost = self.cost_function(trainX, trainY, self.evaluate)
            self.loss_metrics.append(curr_cost)
                
            #If the cost has increased after that iteration, increase lr_pat
            if self.useLearningRateTuning and curr_cost > prev_cost: 
                lr_pat += 1
                      
            db_pat += 1 #increase db_pat

            #print debug information
            if self.debug and db_pat > self.debug_patience: 
                diff = time() - start_time
                cumulative_time += diff
                print(f'Epoch #{e} || Loss :: {curr_cost} || Gradient Mag: {gradientmag} || Avg Time Per Epoch: {cumulative_time / (e + 1):.8f}s')
                db_pat = 0          
                 
            if self.useLearningRateTuning and lr_pat > self.lr_patience:
                #If the loss has INCREASED after lr_pat epochs, decrement the learning rate
                self.learn_rate = max(self.learning_rate * self.lr_decrease, self.lr_min)
                lr_pat = 0
        
    
    def display_fit(self, X, Y, subdivisions: int, rangeX: float):
        """Displays the Model's Fit to a 2D Set of Data

        Args:
            X (_type_): 1 Dimensional Input Data
            Y (_type_): 1 Dimensional Output Data
            subdivisions (int): Number of subdivisions on the graph
            rangeX (float): The Range Of the X-Axis
        """
        
        py.scatter(X, Y)
        X = [ (a / subdivisions) * rangeX for a in range(subdivisions) ]
        Y = [ self.evaluate([a])[0] for a in X ]
        py.plot(X, Y)
        py.show()
        
    def display_loss_metrics(self):
        """Displays Loss Metrics After Training
        """
        py.xlabel("Epoch")
        py.ylabel("Loss")
        py.plot(self.loss_metrics)
        py.show()
    def display_gradient_magnitude_metrics(self):
        """Displays Gradient Magnitude Metrics After Training
        """
        py.xlabel("Epoch")
        py.ylabel("Gradient Magnitude")
        py.plot(self.gradient_mag_metrics)
        py.show()
    
    
    def forward_propagation(self, input):
        #outputs of each layer before an activation function is applied
        outputs = []
        #outputs of each layer after an activation function is applied
        acts = []
        
        #If the input is a list, convert it to a numpy array
        if isinstance(input, list):
          input = np.array(input, dtype=np.float64).reshape((len(input), 1))
        elif len(input.shape) == 1:
            #Add a new axis to the input, making it a column vector and not just a flat list
            input = input[..., np.newaxis]
            
        #The activations also include the inputs to the network
        acts.append(input)
        
        #If the first layer is dense, start at the first layer + 1 (index 1)
        start = 0
        if type(self.layers[0]) is DenseLayer:
            start = 1
            
        for layer in self.layers[start:]:
            typeL = type(layer) #type of the current layer
            
            if typeL is DenseLayer or typeL is ConvolutionLayer:
               outputs.append(layer.process(acts[-1]))
               acts.append(layer.activate(outputs[-1], use_derivative=False))                
               
            elif typeL is FlattenLayer:
                acts.append(layer.process(outputs[-1]))
                outputs.append(acts[-1])
            else:
                #Else a pooling layer (max or avg)
                outputs.append(layer.process(acts[-1]))
                acts.append(outputs[-1])

        return outputs, acts            
    

    def backwards_propagation(self, outputs, acts, cost_deriv, predicted_index):
        
        #Store lists of each of the derivative classes for our paramaters
        dbiases = []
        dweights = []
        dkernels = []
        
        deriv_indices = [
                            #TODO why this shit reversed when we taking length?
                            i for i in range(len(self.layers[:-1])) 
                            if type(self.layers[i]) 
                            is ConvolutionLayer or type(self.layers[i]) is DenseLayer 
                            or type(self.layers[i]) is FlattenLayer
                        ]

        shift = 1
        if type(self.layers[0]) is ConvolutionLayer:
            shift = 0
 
        for deriv_index in deriv_indices:
            #The start of backprop is with the loss function derivative
            values = cost_deriv

            for k in range(self.num_layers - 1, deriv_index - 1, -1):
                typeL = type(self.layers[k])

                if (typeL is DenseLayer and k > deriv_index) or typeL is ConvolutionLayer:
                    act_deriv = self.layers[k].activate(outputs[k - shift], predicted_index, use_derivative=True)

                    if not act_deriv is None:
                         values = np.multiply(act_deriv, values)
                
                if (k > deriv_index + 1 and (typeL is DenseLayer or typeL is FlattenLayer)) or (k > deriv_index and typeL is ConvolutionLayer) or typeL is MaxPoolLayer or typeL is AvgPoolLayer:
                    values = self.layers[k].back_process(values, acts[k])

                    
            typeL = type(self.layers[deriv_index])
            if typeL is DenseLayer or typeL is FlattenLayer:
                dbiases.append(values)
                dweights.append(np.matmul(values, acts[deriv_index + 1 - shift].T))
                
            else:
                #Else its a convolution layer
                kernel, bias = self.layers[deriv_index].derive(values, acts[deriv_index])
                dbiases.append(bias)
                dkernels.append(kernel)
            
        return dweights, dbiases, dkernels                 

   
    def classification_accuracy(self, testX, testY) -> float:
        """Model Classification Accuracy on a Test Set

        Args:
            testX (_type_): Input Set X
            testY (_type_): Output Set Y

        Returns:
            _type_: Accuracy Value in range [0, 1]
        """
        
        count = 0
        for input_set, output_index in zip(testX, testY):
            output = self.evaluate(input_set)
            
            max = 0
            index = 0
            for ind, val in enumerate(output):
                if val > max:
                    max = val
                    index = ind
            
            #+1 if the argmax of both vectors are the same     
            count += 1 if  index == output_index else 0

        return count / len(testY)
     

    def loss_on_dataset(self, testX, testY) -> float:
        """Computes the Model Specific Loss on a Dataset

        Args:
            testX (_type_): Input Set X
            testY (_type_): Output Set Y

        Returns:
            float: The Loss on this Particular Dataset
        """
        sum = 0
        for input_set, output_set in zip(testX, testY):
             output = self.evaluate(input_set)
             sum += self.cost_function(output, output_set)

        return np.sqrt(sum) / len(testY)
    






    def Adam_Optomizer(self, X, Y):
        
        #Construct the average gradient matrices
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.size,1), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
        
        #Stochastic Gradient Descent! Choose a random sample of data points from the data set to choose from
        #The size of this 'mini dataset' will be the batch size
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
            
           #Grab the input and retrieve the outputs and activations 
           #From forward prop needed for back prop
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           #Retrieve the cost derivative needed for backprop
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           
           p_idx = None
           if len(Y.shape) > 1 and Y[0].shape[0] > 1:
                p_idx = argMax(Y[i])
           else:
               p_idx = Y[i]
               
           #Retrieve the gradients
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, predicted_index=p_idx, cost_deriv=cost_deriv)
           
           #Add these gradients to our averages
           for a in range( len(dweights) ):
               avgD_weights[a] += dweights[a]
           for a in range( len(dbiases) ):
               avgD_biases[a] += dbiases[a]
           for a in range( len(dkernels) ):
               avgD_kernels[a] += dkernels[a]
           
        #Then once the mini batch is over, divide these gradients by the batch size    
        for a in range( len(dweights) ):
               avgD_weights[a] /= self.batch_size
        for a in range( len(dbiases) ):
               avgD_biases[a]  /= self.batch_size
        for a in range( len(dkernels) ):
               avgD_kernels[a] /= self.batch_size
        
        
        mag = 0 #Gradient Magnitude
        
        add = 0
        if type(self.layers[0]) is DenseLayer: 
            add = 1
        
            
        #NOW!!! Go through each of the gradients and apply the ADAM optomization to them
        index = 0
        weight_indices = [i + add for i in range(len(self.layers[add:])) if type(self.layers[i]) is DenseLayer]
        for a in range(len(avgD_weights)):
            
            EXPWA_Weight = self.momentum2 * self.prev_EXPWA_Weight[a] + (1 - self.momentum2) * np.square(avgD_weights[a])
            lr_matrix = np.empty(self.layers[weight_indices[index]].weights.shape)
            lr_matrix.fill(self.learning_rate)
            lr_matrix = np.divide(lr_matrix, np.sqrt((EXPWA_Weight + self.epsillon)))
            
            changeW =  self.momentum * self.prev_momentum_Weight[a] + (1 - self.momentum) * avgD_weights[a]
            self.layers[weight_indices[index]].weights -= np.multiply(changeW, lr_matrix)

            index += 1
            self.prev_momentum_Weight[a] = changeW
            self.prev_EXPWA_Weight[a] = EXPWA_Weight
            
            #Add to the gradient magnitude
            if self.debug: mag += np.sum( np.square(changeW) )
          
        #Now onto the biases...
        bias_indices = [i for i in range(len(self.layers)) if (type(self.layers[i]) is DenseLayer and i > 0) or type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for b in range(len(avgD_biases)):
              
            EXPWA_Bias = self.momentum2 * self.prev_EXPWA_Bias[b] + (1 - self.momentum2) * np.square(avgD_biases[b])
            lr_matrix = np.empty(  self.layers[ bias_indices[index] ].biases.shape )
            lr_matrix.fill(self.learning_rate)
            lr_matrix = np.divide(lr_matrix, np.sqrt((EXPWA_Bias + self.epsillon)))
            
            changeB =  self.momentum * self.prev_momentum_Bias[b] + (1 - self.momentum) * avgD_biases[b] 
            self.layers[ bias_indices[index] ].biases -= np.multiply(changeB, lr_matrix)
            index += 1
            self.prev_momentum_Bias[b] = changeB
            self.prev_EXPWA_Bias[b] = EXPWA_Bias
             
            #Add to the gradient magnitude
            if self.debug: mag += np.sum( np.square(changeB) )
         
        #And then the kernel for convolutional layers...
        kernel_indices = [i for i in range(len(self.layers)) if type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for c in range(len(avgD_kernels)):
            
            EXPWA_Kernel = self.momentum2 * self.prev_EXPWA_kernel[c] + (1 - self.momentum2) * np.square(avgD_kernels[c])
            lr_matrix = np.empty(self.layers[kernel_indices[index]].kernels.shape)
            lr_matrix.fill(self.learning_rate)
            lr_matrix = np.divide(lr_matrix, np.sqrt((EXPWA_Kernel + self.epsillon)))
            
            changeK = self.momentum * self.prev_momentum_kernel[c] + (1 - self.momentum) * avgD_kernels[c] 
            self.layers[ kernel_indices[index] ].kernels -= np.multiply(changeK, lr_matrix)
            index += 1
            self.prev_momentum_kernel[c] = changeK
            self.prev_EXPWA_kernel[c] = EXPWA_Kernel
             
            #Add to the gradient magnitude
            if self.debug: mag += np.sum( np.square(changeK) )
            
        return mag




    def Default_Optomizer(self, X, Y): 
        
         #Construct the average gradient matrices
        avgD_weights = [ np.zeros(a.weights.shape, dtype=np.float64) for a in self.layers[1:] if type(a) is DenseLayer ]
        avgD_biases = [ np.zeros((a.biases.shape), dtype=np.float64) for index, a in enumerate(self.layers) if (type(a) is DenseLayer and index > 0) or type(a) is ConvolutionLayer ]
        avgD_kernels = [ np.zeros(a.kernel_shape, dtype=np.float64) for a in self.layers if type(a) is ConvolutionLayer ]
        
        
        #Stochastic Gradient Descent! Choose a random sample of data points from the data set to choose from
        #The size of this 'mini dataset' will be the batch size
        rand_data_points = np.random.randint(0, len(X), size=self.batch_size)
        for i in rand_data_points:
            
           #Grab the input and retrieve the outputs and activations 
           #From forward prop needed for back prop
           inputs = X[i]
           outputs, acts = self.forward_propagation(inputs)
           
           #Retrieve the cost derivative needed for backprop
           cost_deriv =  self.cost_function_derivative(X, Y, data_index=i, output_values = acts[-1])
           
           p_idx = None
           if len(Y.shape) > 1 and Y[0].shape[0] > 1:
                p_idx = argMax(Y[i])
           else:
               p_idx = Y[i]
               
           #Retrieve the gradients
           dweights, dbiases, dkernels = self.backwards_propagation(outputs, acts, predicted_index=p_idx, cost_deriv=cost_deriv)

           #Add these gradients to our averages
           for a in range( len(dweights) ):
               avgD_weights[a] += dweights[a]
           for a in range( len(dbiases) ):
               avgD_biases[a] += dbiases[a]
           for a in range( len(dkernels) ):
               avgD_kernels[a] += dkernels[a]
            
        #Then once the mini batch is over, divide these gradients by the batch size     
        for a in range( len(dweights) ):
               avgD_weights[a] /= self.batch_size
        for a in range( len(dbiases) ):
               avgD_biases[a]  /= self.batch_size
        for a in range( len(dkernels) ):
               avgD_kernels[a] /= self.batch_size
        
        mag = 0

        index = 0
        weight_indices = [i+1 for i in range(len(self.layers[1:])) if type(self.layers[i]) is DenseLayer]
        for a in range( len(avgD_weights) ):
            
            changeW = avgD_weights[a] * self.learning_rate
            self.layers[weight_indices[index]].weights -= changeW
            index += 1

            #Add to the gradient magnitude
            if self.debug: mag += np.sum( np.square(changeW) )
          
        bias_indices = [i for i in range(len(self.layers)) if (type(self.layers[i]) is DenseLayer and i > 0) or type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for b in range( len(avgD_biases) ):  
            
            changeB =  avgD_biases[b] * self.learning_rate
            self.layers[ bias_indices[index] ].biases -= changeB
            index += 1
             
            #Add to the gradient magnitude
            if self.debug: mag += np.sum(np.square(changeB))
         
        kernel_indices = [i for i in range(len(self.layers)) if type(self.layers[i]) is ConvolutionLayer]
        index = 0
        for c in range( len(avgD_kernels) ):
            
            changeK =  avgD_kernels[c] * self.learning_rate
            self.layers[ kernel_indices[index] ].kernels -= changeK
            index += 1
            
            #Add to the gradient magnitude
            if self.debug: mag += np.sum(np.square(changeK))
            
        return mag





  

          
                     


      
    
   