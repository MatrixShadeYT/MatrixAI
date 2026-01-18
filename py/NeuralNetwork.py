import numpy as np
import json
class AI:
    def __init__(self,file):
        self.file = file
        self.load(file)
    def run(self,inputs):
        inputs = inputs
        for i in range(len(self.network)):
            if type(self.network[i]) == str:
                if self.network[i] == 'relu':
                    inputs = relu.forward(inputs)
                elif self.network[i] == 'softmax':
                    inputs = softmax.forward(inputs)
            elif type(self.network[i]) == dict:
                inputs = Layer(self.network[i],inputs)
            else:
                print('ERROR')
        self.output = inputs
        return self.output
    def save(self,file):
        if type(self.network) == list:
            with open(file,'w') as file:
                json.dump(self.network,file,indent=4)
    def load(self,file):
        if type(file) == list:
            self.network = file
        elif type(file) == str:
            with open(file,'r') as file:
                self.network = json.load(file)
        else:
            self.network = [
                {
                    "weights": [0,0,0],
                    "biases": [0,0,0]
                },
                "softmax"
            ]
            print('ERROR ON LOAD')

class Layer:
    def __init__(self,network,inputs):
        self.weights = np.array(network['weights'])
        self.biases = np.array(network['biases'])
        self.output = self.forward(np.array(inputs))
    def forward(self,inputs):
        return np.dot(np.array(inputs),self.weights)+self.biases

# Loss Functions
class Loss:
    def calculate(self,outputs,true):
        return np.mean(self.forward(np.array(outputs), np.array(true)))
class CategoricalCrossEntropy(Loss):
    def forward(self,pred,true):
        pred = np.array(pred)
        true = np.array(true)
        pred_clipped = np.clip(np.array(pred), 1e-7, 1-1e-7)
        if len(true.shape) == 1:
            correct_confidences = pred_clipped[range(len(pred))]
        elif len(true.shape) == 2:
            correct_confidences = np.sum(pred_clipped*true,axis=1)
        return -np.log(correct_confidences)

# Activation Functions
class relu:
    @staticmethod
    def forward(self,inputs):
        return np.maximum(0, np.array(inputs))
class softmax:
    @staticmethod
    def forward(inputs):
        inputs = np.array(inputs)
        probabilities = np.exp(inputs,axis=1,keepdims=True)/np.sum(np.exp(inputs),axis=1,keepdims=True)
        return probabilities