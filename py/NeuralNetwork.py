import numpy, json
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
        inputs = numpy.array(inputs)
        self.weights = network['weights']
        self.biases = network['biases']
        self.output = self.forward(inputs)
    def forward(self,inputs):
        return numpy.dot(inputs*self.weights)+self.biases

class Loss:
    def calculate(self,outputs,true):
        return numpy.mean(self.forward(output, true))

class CategoricalCrossEntropy(Loss):
    def forward(self,pred,true):
        pred_clipped = numpy.clip(pred, 1e-7, 1-1e-7)
        if len(true.shape) == 1:
            correct_confidences = pred_clipped[range(len(pred)),true]
        elif len(true.shape) == 2:
            correct_confidences = np.sum(pred_clipped*true, axis=1)
        return -numpy.log(correct_confidences)

class relu:
    @staticmethod
    def forward(inputs):
        return numpy.max(0,inputs)

class softmax:
    @staticmethod
    def forward(inputs):
        return np.exp(inputs)/np.sum(np.exp(inputs),axis=1,keepdims=True)