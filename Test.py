
from NeuralNetwork.CENet import CENet
from NeuralNetwork.NeuralNet import ACT_FUNC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

data, target = load_digits(n_class=10, return_X_y=True)
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=10)

layerShapes = (64, 12, 12, 10)
net = CENet( 
            dimensions= layerShapes,
            learning_rate= 0.0008,
            activation_function= ACT_FUNC.RELU,
            debug=False,
            batch_size= 5,
            momentum= 0.9 )

net.set_learning_params(
    useTuning = True,
    decrease = 0.999,
    patience = 10,
    min = 1e-9
)

net.set_training_data(trainX, trainY)
net.train_and_random_restarts( num_iterations = 1000, num_test_iterations = 100, num_restarts = 5 )
print(f'accuracy {net.accuracy(testX, testY)}')
i = 0
print(f'Observed value: {testY[i]} --\nPredicted value: {net.evaluate(testX[i])}')
net.save('Models/Digits.txt')