import numpy as np
import json
class AI:
    def __init__(self,network):
        self.sync(network)
    def sync(self,file):
        if type(file) == 'string':
            with open(file,'r') as file:
                self.network = json.load(file)
        elif type(file) == list:
            self.network = []
            for i in file:
                if type(file[i]) == string:
                    if file[i] == 'relu':
                        self.network.append(relu)
                    elif file[i] == 'softmax':
                        self.network.append(softmax)
                    else:
                        print(f'Error: {file[i]}')
                elif type(file[i]) == dict:
                    self.network.append(file[i])
                else:
                    print(f'Error: {file[i]}')
        else:
            with open(file,'w') as file:
                json.dump(self.network,file,indent=4)

class Layer:
    def __init__(self,network,inputs):
        self.weights = network['weights']
        self.biases = network['biases']
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
    def forward(inputs):
        return np.maximum(0, np.array(inputs))
class softmax:
    @staticmethod
    def forward(inputs):
        inputs = np.array(inputs)
        return np.exp(inputs)/np.sum(np.exp(inputs),keepdims=True)