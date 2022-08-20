import math
import random
from cmath import tanh, cosh

class Neuron():
    def __init__(self, activation, numInputs = 0):
        self.bias = random.uniform(-1, 1)
        self.numInputs = numInputs
        self.inputs = [0 for _ in range(numInputs)]
        self.weights = [random.uniform(-1, 1) / math.sqrt(numInputs) for _ in range(numInputs)]
        self.grad = [0 for _ in range(numInputs)]
        self.definitionGrad = [0 for _ in range(numInputs)]
        self.biasGrad = 0
        self.stimulus = 0
        self.activationOptions = {
            "Sigmoid": lambda x: 1 / (1 + math.exp(-1 * x)),
            "Hyp tan": lambda x: tanh(x).real
        }
        self.activationDerivativeOptions = self.initActivationDerivative()
        self.activationToUse = activation
    
    def __str__(self):
        out = "Neuron\n"
        out += "Bias " + str(self.bias) + "\n"
        out += "Weights:\n"
        for weight in self.weights:
            out += str(weight) + "\n"
        return out
    
    def initActivationDerivative(self):
        actDerivative = {}
        
        def sig(x):
            activation = self.activationOptions["Sigmoid"](x)
            return activation * (1 - activation)
        
        actDerivative["Sigmoid"] = sig
        actDerivative["Hyp tan"] = lambda x: 1 / (cosh(x).real ** 2)
        return actDerivative
        
    def activationFunction(self, stimulus):
        return self.activationOptions[self.activationToUse](stimulus)

    def activationDerivative(self, stimulus):
        return self.activationDerivativeOptions[self.activationToUse](stimulus)
    
    def fire(self, inputSignals):
        self.stimulus = 0
        for i, inSig in enumerate(inputSignals):
            self.inputs[i] = inSig
            self.stimulus += inSig * self.weights[i]
        self.stimulus += self.bias
        return self.activationFunction(self.stimulus)
    
    def applyGradients(self, learnRate):
        for i in range(len(self.weights)):
            self.weights[i] -= self.grad[i] * learnRate
        self.bias -= self.biasGrad * learnRate

    def zeroGradients(self):
        self.grad = [0 for _ in range(self.numInputs)]
        self.definitionGrad = [0 for _ in range(self.numInputs)]
        self.biasGrad = 0