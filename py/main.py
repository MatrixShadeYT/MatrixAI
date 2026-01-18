from NeuralNetwork import *
net = [
    {
        'weights': [0.05,0.05,0.05],
        'biases': [0.05,0.05,0.05]
    },
    'softmax'
]
inputs = [1,0,1]
layer = Layer(net[0],inputs)
soft = softmax(layer.output)
print(soft)