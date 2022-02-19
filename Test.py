



from audioop import bias
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
net.add(DenseLayer(2, ACT_FUNC.RELU, weights=weights_one, biases=biases_one))
net.add(DenseLayer(1, ACT_FUNC.NONE, weights=weights_two, biases=biases_two))
net.set_hyper_params(learningRate=0.01, momentum=0.9, EWA=0.9, epsillon=0.0000001, batch_size=2)
net.compile(Optomizer.ADAM, Cost.SQUARE_RESIDUALS, accuracy_type='regressive', debug=True, debug_patience=10)

outputs, acts = net.forward_propagation([1])
#print('OUTPUT ', acts[1])
cost_deriv = SSR_Derivative(X, Y, 0, acts[-1])
dweights, dbiases, dkernels = net.backwards_propagation(outputs, acts, cost_deriv )
print('dweights ',dbiases)


#print('cost derivative ',cost_deriv)
net.fit(X, Y, numIterations=1000)
net.display(X, Y, 50, 1)
#print(net.evaluate([[1]]))