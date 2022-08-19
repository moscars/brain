from cmath import tanh, cosh
import random
import time

class Neuron():
    def __init__(self, numInputs = 0):
        self.bias = random.uniform(-1, 1)
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
        for _ in range(self.size):
            self.neurons.append(Neuron(self.numInNeurons))
        
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
        self.t = 0
        
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

    # Gets loss for one testcase
    # split for each output node
    def getLocalLoss(self, prediction, truth):
        loss = [0 for _ in range(len(prediction))]
        for i in range(len(prediction)):
            loss[i] = self.lossFunction(prediction[i], truth[i])
        return loss

    # Gets total network loss
    def getLoss(self, localLosses):
        return sum([sum(x) for x in localLosses])
    
    # Gets the gradient for each weight and bias by
    # using the definition of the derivative
    def getGradByDefinition(self, xs, ypred, ys):
        h = 0.0001
        for idx in range(len(xs)):
            loss = self.getLocalLoss(ypred[idx], ys[idx])
            totLossf = sum(loss)
            for layerIdx in range(len(self.layers)):
                for neuronIdx in range(self.layers[layerIdx].size):
                    for weightIdx in range(self.layers[layerIdx].neurons[neuronIdx].numInputs):
                        self.layers[layerIdx].neurons[neuronIdx].weights[weightIdx] += h
                        ypred2 = self.classify(xs[idx])
                        self.layers[layerIdx].neurons[neuronIdx].weights[weightIdx] -= h
                        lossh = self.getLocalLoss(ypred2, ys[idx])
                        totLossfplush = sum(lossh)
                        self.layers[layerIdx].neurons[neuronIdx].grad[weightIdx] += (totLossfplush - totLossf) / h

                    
                    self.layers[layerIdx].neurons[neuronIdx].bias += h
                    ypred2 = self.classify(xs[idx])
                    self.layers[layerIdx].neurons[neuronIdx].bias -= h
                    lossh = self.getLocalLoss(ypred2, ys[idx])
                    totLossfplush = sum(lossh)
                    self.layers[layerIdx].neurons[neuronIdx].biasGrad += (totLossfplush - totLossf) / h
    
    
    def backPropagate(self, xs, ys):
        # Gets the weights between layers
        weightsToNextLayer = {}
        for i in range(len(self.layers) - 2, -1, -1):
            weightsToNextLayer[i] = self.getWeightsBetweenLayers(self.layers[i+1], self.layers[i])
        
        # Iterates over each pair of inputs and outputs in the training data
        for idx in range(len(xs)):
            pred = self.classify(xs[idx])
            
            # Holds the derivative of the loss with respect to the stimulus for the neurons
            # in the current layer
            currentLayerGrad = [0 for _ in range(self.layers[-1].size)]
            
            # Finds the gradients for each weight and bias in the output layer
            for i, neuron in enumerate(self.layers[-1].neurons):
                currentLayerGrad[i] = self.lossDerivative(pred[i], ys[idx][i]) * neuron.activationDerivative(neuron.stimulus)
                for widx in range(len(self.layers[-1].neurons[i].weights)):
                    self.layers[-1].neurons[i].grad[widx] += currentLayerGrad[i] * self.layers[-1].neurons[i].inputs[widx]
                    
                self.layers[-1].neurons[i].biasGrad += currentLayerGrad[i]
            
            # Finds the gradients for each weight and bias in the hidden layers
            for i in range(len(self.layers) - 2, -1, -1):
                currentLayerGrad = self.updateGradForCurrentLayer(self.layers[i], self.layers[i+1], currentLayerGrad, weightsToNextLayer[i])
                
    def updateGradForCurrentLayer(self, layer, nextLayer, nextLayerGrad, weightsToNextLayer):
        currentLayerGrad = [0 for _ in range(layer.size)]
        for neuronIdx in range(layer.size):
            # Update dLoss/dStimulusToCurrentNeuron
            for nextNeuronIdx in range(nextLayer.size):
                currentLayerGrad[neuronIdx] += nextLayerGrad[nextNeuronIdx] * weightsToNextLayer[neuronIdx][nextNeuronIdx]
            currentLayerGrad[neuronIdx] *= layer.neurons[neuronIdx].activationDerivative(layer.neurons[neuronIdx].stimulus)
            
            # Update gradient for neuron weights and bias
            for weightIdx in range(layer.neurons[neuronIdx].numInputs):
                layer.neurons[neuronIdx].grad[weightIdx] += currentLayerGrad[neuronIdx] * layer.neurons[neuronIdx].inputs[weightIdx]
            layer.neurons[neuronIdx].biasGrad += currentLayerGrad[neuronIdx]
        return currentLayerGrad
        
    
    def getWeightsBetweenLayers(self, currentLayer, prevLayer):
        s = time.time()
        weightsToNextLayer = [[0 for _ in range(currentLayer.size)] for _ in range(prevLayer.size)]
        for i, neuron in enumerate(currentLayer.neurons):
            for j in range(len(prevLayer.neurons)):
                weightsToNextLayer[j][i] = neuron.weights[j]
        e = time.time()
        self.t += (e - s)
        return weightsToNextLayer
    
    def applyGradients(self, learnRate):
        for layer in self.layers:
            layer.applyGradients(learnRate)
    
    def zeroGradients(self):
        for layer in self.layers:
            layer.zeroGradients()
    
    def printOut(self, out):
        for row in out:
            for num in row:
                print(round(num, 3), end=" ")
            print()

    def learn(self, runs, learnRate, xs, ys, printPred = False):
        for k in range(runs):
            ypred = [self.classify(x) for x in xs]
            
            loss = self.getLoss([self.getLocalLoss(ypred[i], ys[i]) for i in range(len(ypred))])
            #self.getGradByDefinition(xs, ypred, ys)
            self.backPropagate(xs, ys)

            # Backprop
            self.applyGradients(learnRate)
            self.zeroGradients()
            
            print("Iteration: {}, Loss: {}".format(k, loss))
            if printPred:
                self.printOut(ypred)