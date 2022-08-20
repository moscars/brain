from neuron import Neuron

class Layer():
    def __init__(self, numInNeurons, size, activation):
        self.size = size
        self.numInNeurons = numInNeurons
        self.neurons = []
        self.initNeurons(activation)
    
    def __str__(self):
        out = ["Layer with {} inputs and {} outputs:".format(self.numInNeurons, self.size)]
        for neuron in self.neurons:
            out.append(neuron.__str__())
        return '\n'.join(out)
    
    def initNeurons(self, activation):
        for _ in range(self.size):
            self.neurons.append(Neuron(activation, self.numInNeurons))
        
    def classify(self, inputSignals):
        assert(len(inputSignals) == self.numInNeurons)
        return [neuron.fire(inputSignals) for neuron in self.neurons]
    
    def applyGradients(self, learnRate):
        for neuron in self.neurons:
            neuron.applyGradients(learnRate)
    
    def zeroGradients(self):
        for neuron in self.neurons:
            neuron.zeroGradients()