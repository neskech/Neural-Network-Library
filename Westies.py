
from json import load
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from ImageParse.Parser import getData
from NeuralNetwork.Layer.Layer import ACT_FUNC, AvgPoolLayer, ConvolutionLayer, DenseLayer, FlattenLayer, MaxPoolLayer
from NeuralNetwork.Model.Cost import Cost

from NeuralNetwork.Model.NeuralNet import NeuralNet, Optomizer

data, target = load_digits(n_class=10, return_X_y=True)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

net = NeuralNet()
net.add(DenseLayer(size=64))
net.add(DenseLayer(size=30, func= ACT_FUNC.TANH))
net.add(DenseLayer(size=30, func=ACT_FUNC.TANH))
net.add(DenseLayer(10, ACT_FUNC.SOFTMAX))

net.set_hyper_params(learningRate=0.006, momentum=0.80, EWA=0.80, epsillon=0.0000001, batch_size=4)
net.compile(Optomizer.ADAM, Cost.CROSS_ENTROPY, accuracy_type='classification', debug=True, debug_patience=10)

net.fit(trainX, trainY, numIterations=5000, numRestarts=0, numRestart_Iterations=10)
print('hello!',net.accuracy(testX, testY))
print('hello!',net.accuracy(trainX, trainY))


for i in range( len(testX) ) :
    output = net.evaluate(testX[i])
    max = 0
    index = 0
    for ind, val in enumerate(output):
         if val > max:
             max = val
             index = ind
    print(f'Observed value: {testY[i]} --Predicted value: {index}')