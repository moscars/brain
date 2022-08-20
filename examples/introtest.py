import sys 
sys.path.append('..')
from neuralnetwork import Brain

brain = Brain([3, 6, 4, 2], "Hyp tan")
xs = [
    [2.0, 3.0, -1.0],
    [2.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [2.0, 2.0, 0.7]
]

ys = [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]]

brain.learn(2000, 0.01, xs, ys)

print()
print("Final predictions:")
for x, y in zip(xs, ys):
    print([round(pred, 2) for pred in brain.classify(x)], y)