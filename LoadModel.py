
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from NeuralNetwork.Model.Cost import Cost
from NeuralNetwork.Model.Model import Model, Optomizer, AccuracyTP

#8x8 Mnist, 10 output classes
data, target = load_digits(n_class=10, return_X_y=True)
#Normalizing
scaler = StandardScaler().fit(data)
data = scaler.transform(data)

#Validation Split
trainX, testX, trainY, testY = train_test_split(data, target, train_size= 0.80, random_state=16)

net = Model()
#Mnist Basic -- Without convolution
net.load("./SaveModels/MnistBasic.txt")

#No need to set hyper parameters if model is not to be trained again

net.compile(Optomizer.ADAM, Cost.CROSS_ENTROPY, AccuracyTP.CLASSIFICATION, debug=True, debug_patience=10)

#Should be around 99% accuracy on both sets
print(f'Test accuracy {net.accuracy(testX, testY)}')
print(f'Train accuracy {net.accuracy(trainX, trainY)}')





#LOADING CONVOLUTION NET !!! Remove the exit!
exit(1)

#8x8 Mnist, 3 output classes
data, target = load_digits(n_class=3, return_X_y=True)
#Normalizing
scaler = StandardScaler().fit(data)
data = scaler.transform(data)

#Reshaping to from column vector to multidimensional tensor
data = data.reshape( (data.shape[0], 1, 8, 8))

#Validation Split
trainX, testX, trainY, testY = train_test_split(data, target, train_size= 0.80, random_state=16)

net2 = Model()
net2.load("./SaveModels/MnistConv.txt")

#No need to set hyper parameters if model is not to be trained again

net2.compile(Optomizer.ADAM, Cost.CROSS_ENTROPY, AccuracyTP.CLASSIFICATION, debug=True, debug_patience=10)

#Should be around 99% accuracy on both sets
print(f'Test accuracy {net2.accuracy(testX, testY)}')
print(f'Train accuracy {net2.accuracy(trainX, trainY)}')

