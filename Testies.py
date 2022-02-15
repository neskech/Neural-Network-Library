

from NeuralNetwork.Model.CENet import CENet

net = CENet('Models/Digits2.txt')
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np

data, target = load_digits(n_class=10, return_X_y=True)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)
net.set_training_data(trainX,trainY)
print(net.accuracy(testX,testY))