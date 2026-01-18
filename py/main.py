from NeuralNetwork import Layer, relu, softmax, CategoricalCrossEntropy
import matplotlib.pyplot as plt
net = [
    {
        'weights': [0.25,0.25,0.25],
        'biases': [0.25,0.25,0.25]
    },
    'softmax'
]
loss = CategoricalCrossEntropy()
true = [0,1,0]
inputs = [1,0,1]
point = []
for i in range(10):
    layer = Layer(net[0],inputs)
    soft = softmax.forward(layer.output)
    loss = loss.calculate(soft,true)
    point.append(loss)
plt.figure(figsize=(1,3))
plt.xlim(0,12)
plt.ylim(0,1)
plt.scatter([i+0.5 for i in range(len(point))], point)
plt.show()