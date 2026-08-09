import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits() 
norm_data = digits.data / 16



class Layer:
    def __init__(self, w, b, ):
        self.w = np.random.rand(*w) * 0.01 # *w to unload the tuple, input needs 2 numbers
        self.b = np.zeros((b)) 

    def Forward(self, data):
         y = data @ self.w + self.b
         return y
    
def Relu(y):
    y = np.maximum(0,y)
    return y  


def SoftMax(x):
    x = np.exp(x)
    sums = np.sum(x, axis=1, keepdims=True)
    x = x / sums
        
    return x

l1 = Layer((64,32), (1,32))
y1 = l1.Forward(norm_data)

y1 = Relu(y1)

l2 = Layer((32,16), (1,16))
y2 = l2.Forward(y1)

y2 = Relu(y2)

l3 = Layer((16,10), (1,10))
y3 = l3.Forward(y2)

print(SoftMax(y3))

# print(np.shape(y3))

