import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

# plt.matshow(digits.images[7], cmap="gray")
# plt.show()

w = np.random.rand(64,32) * 0.01
print(f"Weights shape {w.shape}")

b = np.zeros((1,32))
print(f"Bias shape {b.shape}")

y = (digits.data / 16) @ w + b

relu = np.maximum(0,y)
print(f"fOWARD PASS output shape {relu.shape}")


# Weights shape (64, 32)
# Bias shape (1, 32)
# fOWARD PASS output shape (1797, 32)