

from NeuralNetwork.Model.SSRNet import SSRNet
from NeuralNetwork.Model.NeuralNet import ACT_FUNC
from sklearn.model_selection import train_test_split

data = [ [0], [0.5], [1], [1.5] ]
target = [0, 1, 0, 1]
trainX, testX, trainY, testY = train_test_split(data,target, train_size= 0.80, random_state=16)

layerShapes = (1, 4, 4,  1)
net = SSRNet( 
            dimensions= layerShapes,
            learning_rate= 0.01,
            activation_function= ACT_FUNC.RELU,
            debug=True,
            batch_size= 4,
            momentum= 0.90,
            EXPWA= 0.90,
            epsillon=0.0000001)

net.set_learning_params(
    useTuning = False,
    decrease = 0.90,
    patience = 50,
    min = 1e-9
)

net.use_last_activation = True
net.set_training_data(trainX, trainY)
net.train_and_random_restarts( num_iterations = 3000, num_test_iterations = 100, num_restarts = 5)
print(f'accuracy {net.accuracy_regress(testX, testY)}')

for i in range( len(testX) ) :
    print(f'Observed value: {testY[i]} --Predicted value: {net.evaluate(testX[i])}')
net.display_data_and_fit(2,50)