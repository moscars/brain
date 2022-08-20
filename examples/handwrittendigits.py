import sys 
sys.path.append('..')
from neuralnetwork import Brain
# Create dataset
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

def getYVals(prevY):
    newY = []
    for digit in prevY:
        desiredOutput = [0 for _ in range(10)]
        desiredOutput[digit] = 1
        newY.append(desiredOutput)
    return newY

def getXVals(prevX):
    newX = []
    for x in prevX:
        newInput = x.flatten()
        newInput = list(map(lambda x: x / 255, newInput))
        newX.append(newInput)
    return newX

brain = Brain([784, 10], "Sigmoid")

xIn = getXVals(x_train)
yIn = getYVals(y_train)

brain.learn(100, 0.003, xIn, yIn, 50)