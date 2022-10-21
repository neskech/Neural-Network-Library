
from enum import Enum
from NeuralNetwork.Model import Activations as acts


class ACT_FUNC(Enum):
    SOFT_PLUS = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4
    NONE = 5
    
def activation_to_str(act: ACT_FUNC) -> str:
    l = ["Soft Plus", "Relu", "Sigmoid", "Tanh", "Softmax", "None"]
    return l[act.value]

def str_to_activation(act: str) -> ACT_FUNC:
    match act:
        case "Soft Plus":
            return ACT_FUNC.SOFT_PLUS
        case "Relu":
            return ACT_FUNC.RELU
        case "Sigmoid":
            return ACT_FUNC.SIGMOID
        case "Tanh":
            return ACT_FUNC.TANH
        case "Softmax":
            return ACT_FUNC.SOFTMAX
        case "None":
            return ACT_FUNC.NONE

def parse_tuple(string: str):
    string = string[1:-1]
    
    tup = ()
    
    a = 0
    while a < len(string):
        start = a
        while a < len(string) and string[a] != ',':
            a += 1

        tup += (int(string[start:a]),)
        a += 2

    return tup
        

#BASE CLASS
class Layer:
    def __init__(self, size : int, func : ACT_FUNC) -> None:
        self.size = size
        self.activation_func = func
        self.output_shape = None
        self.input_shape = None
        self.setActivation(func)
    
    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###
    #OVERWRITTEN METHODS
    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###
    
    def process(self, inputs):
        pass
    
    def set_input_size(self, layer):
        pass
    
    def init_rand_params(self, seed : int, mean : float = 0, SD : float = 1):
        pass
    
    def back_process(self, inputs, inputs_two):
        pass
    
    def save_str(self) -> str:
        pass
    
    def load(self, str):
        pass
    
    def activate(self, inputs, predicted_index = -1, use_derivative : bool = False):
         if self.activation is None and use_derivative == False:
             return inputs
         if self.activation_derivative is None and use_derivative:
             return None
         elif use_derivative and predicted_index == -1:
             raise Exception('ERROR: You\'re using the derivative but have a pred index of -1')
         if not use_derivative:
             #predicted index for softmax and cross entropy
             return self.activation(inputs, predicted_index)
         else:
             #predicted index for softmax and cross entropy
             return self.activation_derivative(inputs, predicted_index)
         
    def setActivation(self, func : ACT_FUNC):
        #Activation Function Function Pointers
        if func is ACT_FUNC.RELU:
            self.activation = acts.Relu
            self.activation_derivative = acts.Relu_Deriv

        elif func is ACT_FUNC.SOFT_PLUS:
            self.activation = acts.softPlus
            self.activation_derivative = acts.softPlus_Deriv

        elif func is ACT_FUNC.SIGMOID:
            self.activation = acts.sigmoid
            self.activation_derivative = acts.sigmoid_Deriv

        elif func is ACT_FUNC.TANH:
            self.activation = acts.hyperbolic_tangent
            self.activation_derivative = acts.hyperbolic_tangent_Deriv
            
        elif func is ACT_FUNC.SOFTMAX:
            self.activation = acts.softMax
            self.activation_derivative = acts.softMax_Deriv
                  
        else:
           self.activation = None
           self.activation_derivative = None
           # raise Exception('ERROR: Unknown / unsupported Activation Function')


      