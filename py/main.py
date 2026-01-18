from NeuralNetwork import AI
import matplotlib.pyplot as plt

network = AI('network.json')
print(network.run([1,0,1]))

def render():
    plt.figure(figsize=(1,3))
    plt.xlim(-1,10)
    plt.ylim(-1,10)
    plt.scatter(range(len(point)),point)
    plt.show()