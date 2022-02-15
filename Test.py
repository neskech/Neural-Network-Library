
from statistics import mean
from xml.sax import xmlreader
from NeuralNetwork.Model.CENet import CENet
from NeuralNetwork.Model.NeuralNet import ACT_FUNC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np

data, target = load_digits(n_class=10, return_X_y=True)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

layerShapes = (64, 25, 25, 10)
net = CENet( 
            dimensions= layerShapes,
            learning_rate= 0.01,
            activation_function= ACT_FUNC.RELU,
            debug=True,
            batch_size= 5,
            momentum= 0.90,
            EXPWA= 0.90,
            epsillon=0.0000001)

net.set_learning_params(
    useTuning = False,
    decrease = 0.90,
    patience = 50,
    min = 1e-9
)

net.use_last_activation = False
net.set_training_data(trainX, trainY)
net.train_and_random_restarts( num_iterations = 5000, num_test_iterations = 100, num_restarts = 5)
print(f'accuracy {net.accuracy(testX, testY)}')

for i in range( len(testX) ) :
    print(f'Observed value: {testY[i]} --Predicted value: {net.evaluate(testX[i])}')
net.save('Models/Digits2.txt')