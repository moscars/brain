import random
from layer import Layer
    
class Brain():
    def __init__(self, sizes, activation):
        self.layers = []
        self.sizes = sizes
        self.initLayers(activation)
        
    def __str__(self):
        out = ["Brain: "]
        for layer in self.layers:
            out.append(layer.__str__())
        return '\n'.join(out)
            
    def initLayers(self, activation):
        inputs = self.sizes[0]
        for i in range(1, len(self.sizes)):
            outputs = self.sizes[i]
            self.layers.append(Layer(inputs, outputs, activation))
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
    # using the definition of the derivative (Only used for testing)
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
    
    # Get the gradients for each parameter more efficently with the backpropagation algorithm
    def backpropagate(self, xs, ys):
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
        weightsToNextLayer = [[0 for _ in range(currentLayer.size)] for _ in range(prevLayer.size)]
        for i, neuron in enumerate(currentLayer.neurons):
            for j in range(len(prevLayer.neurons)):
                weightsToNextLayer[j][i] = neuron.weights[j]
        return weightsToNextLayer
    
    def applyGradients(self, learnRate):
        for layer in self.layers:
            layer.applyGradients(learnRate)
    
    def zeroGradients(self):
        for layer in self.layers:
            layer.zeroGradients()

    def learn(self, epochs, learnRate, xs, ys, batchSize = None):
        for epoch in range(epochs):
            if batchSize is None:
                # learn on all examples each run
                batchSize = len(xs)
            
            numBatchRuns = len(xs) // batchSize
            hundredLastAverage = []
            
            combine = list(zip(xs, ys))
            random.shuffle(combine)
            
            # Uses batching to increase the learning speed
            # The network learns on a small set of examples
            # and thus gets to update its parameters more often
            # although with more noisy gradients
            for batchRun in range(numBatchRuns):
                sampleXs = []
                sampleYs = []
                for index in range(batchRun * batchSize, min(batchRun * batchSize + batchSize, len(combine))):
                    sampleXs.append(combine[index][0])
                    sampleYs.append(combine[index][1])
                    
                ypred = [self.classify(x) for x in sampleXs]
                
                loss = self.getLoss([self.getLocalLoss(ypred[i], sampleYs[i]) for i in range(len(ypred))])
                
                self.backpropagate(sampleXs, sampleYs)

                # Backprop
                self.applyGradients(learnRate)
                self.zeroGradients()
                
                accuracy = int(self.getCurrentAccuracy(ypred, sampleYs) * 100)
                hundredLastAverage.append(accuracy)
                if(len(hundredLastAverage) > 100):
                    hundredLastAverage.pop(0)
                print("Iteration: {} {}, Loss: {}".format(epoch, batchRun, loss))
                print("Accuracy this batch: {}%, Average last 100: {}%".format(accuracy, int(sum(hundredLastAverage) / len(hundredLastAverage))))
    
    def getCurrentAccuracy(self, ypred, ys):
        correct = 0
        for yp, yc in zip(ypred, ys):
            largestIndex = 0
            #check which output neuron fires the strongest
            largest = 0
            truthIndex = 0
            for i in range(len(yp)):
                if yp[i] > largest:
                    largest = yp[i]
                    largestIndex = i
                if yc[i] == 1:
                    truthIndex = i
            if largestIndex == truthIndex:
                correct += 1
        
        return correct / len(ypred)
            