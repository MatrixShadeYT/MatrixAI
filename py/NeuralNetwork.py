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
    def __init__(self,network):
        self.weights = network['weights']
        self.biases = network['biases']
    def forward(self,inputs):
        output = numpy.dot(inputs*self.weights)+self.biases
        return output

class relu:
    @staticmethod
    def forward(inputs):
        return numpy.max(0,inputs)

class softmax:
    @staticmethod
    def forward(inputs):
        return np.exp(inputs)/np.sum(np.exp(inputs),axis=1,keepdims=True)