from cmath import tanh, cosh
import random

class Neuron():
    def __init__(self, idx, bias = 0, numInputs = 0):
        self.id = idx
        self.bias = bias
        self.numInputs = numInputs
        self.inputs = [0 for _ in range(numInputs)]
        self.weights = [random.uniform(-1, 1) for _ in range(numInputs)]
        self.grad = [0 for _ in range(numInputs)]
        self.definitionGrad = [0 for _ in range(numInputs)]
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
        return 1 / (cosh(stimulus).real ** 2)
    
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
        seed = random.randint(0, 10000)
        for k in range(self.size):
            self.neurons.append(Neuron(seed + k, 0, self.numInNeurons))
        
    def classify(self, inputSignals):
        assert(len(inputSignals) == self.numInNeurons)
        return [neuron.fire(inputSignals) for neuron in self.neurons]
    
    def applyGradients(self, learnRate):
        for neuron in self.neurons:
            neuron.applyGradients(learnRate)
    
    def zeroGradients(self):
        for neuron in self.neurons:
            neuron.zeroGradients()
    
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

    def getLoss(self, prediction, truth):
        loss = [0 for _ in range(len(prediction))]
        for i in range(len(prediction)):
            loss[i] = self.lossFunction(prediction[i], truth[i])
        
        return loss
    
    def getGradByDefinition(self, xs, ypred, ys):
        h = 0.0001
        totLoss = 0
        for idx in range(len(xs)):
            loss = self.getLoss(ypred[idx], ys[idx])
            totLossf = self.sumLossOverAllXs(loss)
            totLoss += totLossf
            for layerIdx in range(len(self.layers)):
                for neuronIdx in range(self.layers[layerIdx].size):
                    for weightIdx in range(self.layers[layerIdx].neurons[neuronIdx].numInputs):
                        self.layers[layerIdx].neurons[neuronIdx].weights[weightIdx] += h
                        ypred2 = self.classify(xs[idx])
                        self.layers[layerIdx].neurons[neuronIdx].weights[weightIdx] -= h
                        lossh = self.getLoss(ypred2, ys[idx])
                        totLossfplush = self.sumLossOverAllXs(lossh) #definitionGrad
                        self.layers[layerIdx].neurons[neuronIdx].grad[weightIdx] += (totLossfplush - totLossf) / h

                    
                    self.layers[layerIdx].neurons[neuronIdx].bias += h
                    ypred2 = self.classify(xs[idx])
                    self.layers[layerIdx].neurons[neuronIdx].bias -= h
                    lossh = self.getLoss(ypred2, ys[idx])
                    totLossfplush = self.sumLossOverAllXs(lossh)
                    self.layers[layerIdx].neurons[neuronIdx].biasGrad += (totLossfplush - totLossf) / h

        return totLoss
    
    def backPropagate(self, xs, ypred, ys):
        for idx in range(len(xs)):
            self.classify(xs[idx])
            dpWeights = self.getWeightsBetweenLayers(self.layers[-1], self.layers[-2])
            DP = [0 for _ in range(self.layers[-1].size)]
            for i, neuron in enumerate(self.layers[-1].neurons):
                DP[i] = self.lossDerivative(ypred[idx][i], ys[idx][i]) * neuron.activationDerivative(neuron.stimulus)
                for widx in range(len(self.layers[-1].neurons[i].weights)):
                    self.layers[-1].neurons[i].grad[widx] += DP[i] * self.layers[-1].neurons[i].inputs[widx]
                    
                self.layers[-1].neurons[i].biasGrad += DP[i]
            
            for i in range(len(self.layers) - 2, -1, -1):
                DP2 = [0 for _ in range(self.layers[i].size)]
                for j in range(len(self.layers[i].neurons)):
                    for k in range(self.layers[i+1].size):
                        DP2[j] += DP[k] * dpWeights[j][k]
                    DP2[j] *= self.layers[i].neurons[j].activationDerivative(self.layers[i].neurons[j].stimulus)
                    
                    # Update grad for neuron weights
                    for k in range(self.layers[i].neurons[j].numInputs):
                        self.layers[i].neurons[j].grad[k] += DP2[j] * self.layers[i].neurons[j].inputs[k]
                    self.layers[i].neurons[j].biasGrad += DP2[j]
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
            layer.applyGradients(learnRate)
    
    def zeroGradients(self):
        for layer in self.layers:
            layer.zeroGradients()
            
    def sumLossOverAllXs(self, loss):
        tot = 0
        for i in range(len(loss)):
            tot += loss[i]
        return tot
    
    def printOut(self, out):
        for row in out:
            for num in row:
                print(round(num, 3), end=" ")
            print()

    def learn(self, runs, xs, ys):
        for _ in range(runs):
            ypred = [self.classify(x) for x in xs]
            
            totLoss = self.getGradByDefinition(xs, ypred, ys)
                
            self.backPropagate(xs, ypred, ys)

            self.applyGradients(0.05)
            self.zeroGradients()
            
            #print(self.sumLossOverAllXs(loss))
            print(totLoss)
            self.printOut(ypred)
            
#l1 = Layer(2, 3)
#print(l1)
#print(l1.classify([1, 2]))

b = Brain([3, 3, 2])
xs = [
    [2.0, 3.0, -1.0],
    [2.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [2.0, 2.0, 0.7]
]

ys = [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]]

b.learn(5000, xs, ys)