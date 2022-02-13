
from statistics import mean
from xml.sax import xmlreader
from NeuralNetwork.CENet import CENet
from NeuralNetwork.NeuralNet import ACT_FUNC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np

data, target = load_iris(return_X_y=True)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

layerShapes = (4, 8, 8, 3)
net = CENet( 
            dimensions= layerShapes,
            learning_rate= 0.005,
            activation_function= ACT_FUNC.SOFT_PLUS,
            debug=True,
            batch_size= 10,
            momentum= 0.95 )

net.set_learning_params(
    useTuning = True,
    decrease = 0.99,
    patience = 3,
    min = 1e-9
)

net.use_last_activation = False
net.set_training_data(trainX, trainY)
net.train_and_random_restarts( num_iterations = 10000, num_test_iterations = 50, num_restarts = 5)
print(f'accuracy {net.accuracy(testX, testY)}')

for i in range( len(testX) ) :
    print(f'Observed value: {testY[i]} --Predicted value: {net.evaluate(testX[i])}')
net.save('Models/Iris.txt')