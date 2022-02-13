

from NeuralNetwork.CENet import CENet

from sklearn.model_selection import train_test_split
data = [ [0], [0.1], [0.2], [0.3], [0.4] , [0.5], [0.6], [0.7], [0.8], [0.9], [1.0] ]
target = [ 1, 1, 1, 1, 1, 0, 0 ,0 ,0 ,0, 0 ]
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=10)

net = CENet('Models/Digits.txt')
net.set_training_data(trainX,trainY)
print(f'accuracy {net.accuracy(testX, testY)}')
i = 0
print(f'Observed value: {testY[i]} --Predicted value: {net.evaluate([-199])}')
net.save('Models/Digits.txt')