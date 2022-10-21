
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from NeuralNetwork.Layer.Layer import ACT_FUNC
from NeuralNetwork.Layer.Dense import DenseLayer
from NeuralNetwork.Model.Cost import Cost
from NeuralNetwork.Model.Model import Model, Optomizer, AccuracyTP

#8x8 Mnist
data, target = load_digits(n_class=10, return_X_y=True)
#Normalizing
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
#Validation Split
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

#Create the Model
net = Model()
net.add(DenseLayer(size=64))
net.add(DenseLayer(size=30, func=ACT_FUNC.TANH))
net.add(DenseLayer(size=30, func=ACT_FUNC.TANH))
net.add(DenseLayer(10, ACT_FUNC.SOFTMAX))

net.set_hyper_params(learning_rate=0.006, batch_size=4, momentum=0.80, momentum2=0.80, epsillon=0.0000001)
net.compile(Optomizer.ADAM, Cost.CROSS_ENTROPY, AccuracyTP.CLASSIFICATION, debug=True, debug_patience=10)
print(trainY[101], "HAHHAHHAHAHA")
net.train(trainX, trainY, epochs=5000)

print(f'Training accuracy {net.accuracy(trainX, trainY)}')
print(f'Test accuracy {net.accuracy(testX, testY)}')

input("Click enter to display loss metrics...")
net.display_loss_metrics()
input("Click enter to display gradient metrics...")
net.display_gradient_magnitude_metrics()
input("Click enter to see example runs of model performance in action...")

#Example Runs to See Model Performance in Action
for i in range(len(testX)) :
    output = net.evaluate(testX[i])
    max = 0
    index = 0
    for ind, val in enumerate(output):
         if val > max:
             max = val
             index = ind
    print(f'Observed value: {testY[i]} --Predicted value: {index}')
    