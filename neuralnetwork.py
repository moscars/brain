from cmath import tanh
import random

class Neuron():
    def __init__(self, bias = 0, numInputs = 0):
        self.bias = bias
        self.inputs = numInputs
        self.weights = [random.uniform(-1, 1) for _ in range(numInputs)]
    
    def __str__(self):
        out = "Neuron\n"
        out += "Bias " + str(self.bias) + "\n"
        out += "Weights:\n"
        for weight in self.weights:
            out += str(weight) + "\n"
        return out
        
    def activationFunction(self, stimulus):
        return tanh(stimulus).real
    
    def fire(self, inputSignals):
        stimulus = 0
        for i, inSig in enumerate(inputSignals):
            stimulus += inSig * self.weights[i]
        stimulus += self.bias
        return self.activationFunction(stimulus)

class Layer():
    def __init__(self, numInNeurons, size):
        self.size = size
        self.numInNeurons = numInNeurons
        self.neurons = []
        self.initNeurons()
    
    def __str__(self):
        out = ["Layer with {} inputs and {} outputs:".format(self.numInNeurons, self.size)]
        for neuron in self.neurons:
            out.append(neuron.__str__())
        return '\n'.join(out)
    
    def initNeurons(self):
        for _ in range(self.size):
            self.neurons.append(Neuron(0, self.numInNeurons))
        
    def classify(self, inputSignals):
        assert(len(inputSignals) == self.numInNeurons)
        return [neuron.fire(inputSignals) for neuron in self.neurons]
    
class Brain():
    def __init__(self, sizes):
        self.layers = []
        self.sizes = sizes
        self.initLayers()
        
    def __str__(self):
        
        out = ["Brain: "]
        for layer in self.layers:
            out.append(layer.__str__())
        return '\n'.join(out)
            
    def initLayers(self):
        inputs = self.sizes[0]
        for i in range(1, len(self.sizes)):
            outputs = self.sizes[i]
            self.layers.append(Layer(inputs, outputs))
            inputs = outputs
    
    def classify(self, inputSignals):
        feedForward = inputSignals
        for layer in self.layers:
            feedForward = layer.classify(feedForward)
        return feedForward
    
#l1 = Layer(2, 3)
#print(l1)
#print(l1.classify([1, 2]))

b = Brain([2, 3, 3, 2])
print(b)
print(b.classify([1, 2]))
