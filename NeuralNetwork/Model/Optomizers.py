from enum import Enum
import numpy as np

from NeuralNetwork.Layer.Layer import ConvolutionLayer, DenseLayer

class Optomizer(Enum):
    DEFAULT = 0
    ADAM = 1


