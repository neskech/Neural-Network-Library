



from NeuralNetwork.Layer.Layer import ACT_FUNC
from NeuralNetwork.Layer.Dense import DenseLayer
from NeuralNetwork.Model.Cost import Cost
from NeuralNetwork.Model.Model import Model, Optomizer, AccuracyTP

#Custom Data
trainX = [0, 0.5, 1, 2]
trainY = [0, 1, 0, 1]

#Create the Model
net = Model()
net.add(DenseLayer(1))
net.add(DenseLayer(6, ACT_FUNC.TANH))
net.add(DenseLayer(6, ACT_FUNC.TANH))
net.add(DenseLayer(1, ACT_FUNC.NONE))

net.set_hyper_params(learning_rate=0.006, batch_size=1, momentum=0.80, momentum2=0.80, epsillon=0.0000001)
net.compile(Optomizer.ADAM, Cost.SQUARE_RESIDUALS, AccuracyTP.REGRESSION, debug=True, debug_patience=10)
net.train(trainX, trainY, epochs=500)

net.display_fit(trainX, trainY, 50, 2)

input("Click enter to display loss metrics...")
net.display_loss_metrics()
input("Click enter to display gradient metrics...")
net.display_gradient_magnitude_metrics()
input("Click enter to see example runs of model performance in action...")