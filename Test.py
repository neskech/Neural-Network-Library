
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from NeuralNetwork.Layer.Layer import ACT_FUNC, AvgPoolLayer, ConvolutionLayer, DenseLayer, FlattenLayer, MaxPoolLayer
from NeuralNetwork.Model.Cost import Cost

from NeuralNetwork.Model.NeuralNet import NeuralNet, Optomizer

data, target = load_digits(n_class=2, return_X_y=True)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
data = data.reshape( (data.shape[0], 1, 8, 8))
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

net = NeuralNet()
net.add(ConvolutionLayer(num_kernels=4, func= ACT_FUNC.RELU, kernel_shape=(2,2), input_shape=(1,8,8), stride=1))
net.add(MaxPoolLayer(shape=(2,2), stride=2))
net.add(MaxPoolLayer(shape=(2,2), stride=1))
net.add(FlattenLayer())
net.add(DenseLayer(size=5, func= ACT_FUNC.RELU))
net.add(DenseLayer(size=5, func= ACT_FUNC.RELU))
net.add(DenseLayer(3, ACT_FUNC.SOFTMAX))
print([a.input_shape for a in net.layers])
print([a.output_shape for a in net.layers])

net.set_hyper_params(learningRate=0.01, momentum=0.98, EWA=0.98, epsillon=0.00000001, batch_size=3)
net.set_learningRate_settings(patience = 50, decrease= 0.5, min = 1e-7)
net.compile(Optomizer.ADAM, Cost.CROSS_ENTROPY, accuracy_type='classification', debug=True, debug_patience=50)

net.fit(trainX, trainY, numIterations=1000, numRestarts=0, numRestart_Iterations=10)
print(net.accuracy(testX, testY))
print(net.accuracy(trainX,trainY))
print(net.layers[0].kernels)


for i in range( len(testX) ) :
    output = net.evaluate(testX[i])
    max = 0
    index = 0
    for ind, val in enumerate(output):
         if val > max:
             max = val
             index = ind
    print(f'Observed value: {testY[i]} --Predicted value: {index}')