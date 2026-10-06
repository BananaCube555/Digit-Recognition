import numpy as np
from sklearn.datasets import load_digits


class Layer:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, data):
        return data @ self.w + self.b
    
def relu(y):
    y = np.maximum(0,y)
    return y  

#Turn raw data to probabilities
def softMax(x):
    largestnum_x = np.max(x, axis=1, keepdims=True)
    x = x - largestnum_x
    x = np.exp(x)
    sums = np.sum(x, axis=1, keepdims=True)
    x = x / sums
        
    return x
    
# Runs the network

def forward_pass(l1,l2,l3, data):
    y1 = l1.forward(data)
    y1 = relu(y1)


    y2 = l2.forward(y1)
    y2 = relu(y2)

    y3 = l3.forward(y2)
    probs = softMax(y3)

    return y1, y2, y3, probs

