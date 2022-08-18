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
    def __init__(self, size, numInNeurons):
        self.size = size
        self.numInNeurons = numInNeurons
        self.neurons = []
        self.initNeurons()
    
    def __str__(self):
        out = []
        for neuron in self.neurons:
            out.append(neuron.__str__())
        return '\n'.join(out)
    
    def initNeurons(self):
        for _ in range(self.size):
            self.neurons.append(Neuron(0, self.numInNeurons))
        
    def classify(self, inputSignals):
        assert(len(inputSignals) == self.numInNeurons)
        return [neuron.fire(inputSignals) for neuron in self.neurons]
    

l1 = Layer(3, 2)
print(l1)
print(l1.classify([1, 2]))
