import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

plt.matshow(digits.images[10], cmap="gray")
plt.show()

w = np.random.rand(64,32) * 0.01


b = np.zeros((1,32))

y = digits.data @ w + b

print(y.shape)

relu = np.maximum(0,y)

print(relu)
