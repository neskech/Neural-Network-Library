



from audioop import bias

from sqlalchemy import desc
from NeuralNetwork.Layer.Layer import ACT_FUNC, DenseLayer
from NeuralNetwork.Model.Cost import Cost, SSR_Derivative
from NeuralNetwork.Model.NeuralNet import NeuralNet, Optomizer
import numpy as np


X = [ [0], [0.5], [1] ]
Y = [0, 1, 0]
#X = [[1]]
#Y = [5]
weights_one = np.array( [ [1], [2] ], dtype=np.float64 )
weights_two = np.array( [ [3,4] ], dtype=np.float64 )
biases_one = np.array( [ [0], [1] ], dtype=np.float64 )
biases_two = np.array( [ [2] ], dtype=np.float64 )
net = NeuralNet()
net.add(DenseLayer(1))
net.add(DenseLayer(10, ACT_FUNC.TANH))
net.add(DenseLayer(4, ACT_FUNC.TANH))
net.add(DenseLayer(1, ACT_FUNC.NONE))
net.set_hyper_params(learningRate=0.01, momentum=0.99, EWA=0.99, epsillon=0.0000001, batch_size=2)
net.set_learningRate_settings(patience = 50, decrease= 0.5, min = 1e-7)
net.compile(Optomizer.ADAM, Cost.SQUARE_RESIDUALS, accuracy_type='regressive', debug=True, debug_patience=10)



#print('cost derivative ',cost_deriv)
net.fit(X, Y, numIterations=2000)
net.display(X, Y, 50, 1)
#print(net.evaluate([[1]]))