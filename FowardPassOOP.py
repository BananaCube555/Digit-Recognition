import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
norm_data = digits.data / 16

class Layer:
    def __init__(self, W, B): #W shape data(64,32) ,  B (1, 32)

        self.W = np.random.rand(*W) * 0.01
        self.B = np.zeros((B))

    


    def forward(self, Xdata):
        y = (Xdata @ self.W + self.B)
        y =  (np.maximum(0, y))
        return y
        



l1 = Layer((64,32) , (1,32))
y = l1.forward(norm_data)
# print(l1.forward(norm_data).shape)

l2 = Layer((32,16), (1, 16))


y2 = l2.forward(y)
