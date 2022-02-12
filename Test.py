

from NeuralNetwork.SSRNet import SSRNet
from NeuralNetwork.NeuralNet import ACT_FUNC
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

X, Y = load_boston(return_X_y=True)
X = X[ : 30, 2 ]
Y = Y[ : 30 ]

trainX, trainY, testX, testY = train_test_split(X,Y, train_size=0.80, random_state=2)
x = []
y = []
X = []
Y = []
for a, b, c, d in zip(trainX, trainY, testX, testY):
    x.append( [a] )
    y.append( b )
    X.append( [c] )
    Y.append( d )
    
net = SSRNet(fileName='Models/SEXXXXXXXXXXX.txt')
print(net.biases)
net.set_training_data(x,y)
net.display_data_and_fit(10,50)