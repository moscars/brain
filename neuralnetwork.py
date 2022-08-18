from cmath import tanh, cosh
import enum
import random

class Neuron():
    def __init__(self, bias = 0, numInputs = 0):
        self.bias = bias
        self.numInputs = numInputs
        self.inputs = [0 for _ in range(numInputs)]
        self.weights = [random.uniform(-1, 1) for _ in range(numInputs)]
        self.grad = [0 for _ in range(numInputs)]
        self.biasGrad = 0
        self.stimulus = 0
    
    def __str__(self):
        out = "Neuron\n"
        out += "Bias " + str(self.bias) + "\n"
        out += "Weights:\n"
        for weight in self.weights:
            out += str(weight) + "\n"
        return out
        
    def activationFunction(self, stimulus):
        return tanh(stimulus).real

    def activationDerivative(self, stimulus):
        return 1 / (cosh(stimulus) ** 2)
    
    def fire(self, inputSignals):
        self.stimulus = 0
        for i, inSig in enumerate(inputSignals):
            self.stimulus += inSig * self.weights[i]
        self.stimulus += self.bias
        return self.activationFunction(self.stimulus)

    def learn(self):
        for i in range(len(self.weights)):
            self.weights[i] += self.grad[i]
        self.bias += self.biasGrad
    
    def applyGradients(self, learnRate):
        for i, weight in enumerate(self.weights):
            weight += self.grad[i] * learnRate
        self.bias += self.biasGrad * learnRate
    
    
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
    
    def applyGradients(self, learnRate):
        for neuron in self.neurons:
            neuron.applyGradient(learnRate)
    
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
    
    def lossFunction(self, prediction, truth):
        return (prediction - truth) ** 2
    
    def lossDerivative(self, prediction, truth):
        return 2 * (prediction - truth)

    def getLoss(self, predictions, truth):
        loss = [self.lossFunction(predictions[i], truth[i]) for i in range(len(predictions))]
        return loss
    
    def backPropagate(self, predictions, loss):
        dpWeights = self.getWeightsBetweenLayers(self.layers[-1], self.layers[-2])
        DP = [0 for _ in range(self.layers[-1].size)]
        for i, neuron in enumerate(self.layers[-1].neurons):
            DP[i] = self.lossDerivative(predictions[i], loss[i]) * neuron.activationDerivative(neuron.stimulus)
        
        for i in range(len(self.layers) - 2, -1, -1):
            DP2 = [0 for _ in range(self.layers[i].size)]
            for j, neuron in enumerate(self.layers[i].neurons):
                for k in range(self.layers[i+1].size):
                    DP2[j] += DP[k] * dpWeights[j][k]
                DP2[j] *= neuron.activationDerivative(neuron.stimulus)
                
                # Update grad for neuron weights
                for k in range(neuron.numInputs):
                    neuron.grad[k] = DP2[j] * neuron.inputs[k]
                neuron.biasGrad = DP2[j]
            DP = DP2
            
            if i > 0:
                dpWeights = self.getWeightsBetweenLayers(self.layers[i], self.layers[i-1])
    
    def getWeightsBetweenLayers(self, currentLayer, prevLayer):
        dpWeights = [[0 for _ in range(currentLayer.size)] for _ in range(prevLayer.size)]
        for i, neuron in enumerate(currentLayer.neurons):
            for j in range(len(prevLayer.neurons)):
                dpWeights[j][i] = neuron.weights[j]
        return dpWeights
    
    def applyGradients(self, learnRate):
        for layer in self.layers:
            layer.applyGradient(learnRate)
    
    def train(self, runs, xs, ys):
        loss = []
        for _ in range(runs):
            ypred = [self.classify(x) for x in xs]
            loss = self.getLoss(ypred, ys)
            print(sum(loss), loss)

            self.backPropagate(ypred, loss)

            self.applyGradients(0.1)
        
        print(sum(loss), ys, ypred)            
            
    
    
#l1 = Layer(2, 3)
#print(l1)
#print(l1.classify([1, 2]))