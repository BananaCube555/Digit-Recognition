import numpy as np
from forward import Layer,forward_pass  
from sklearn.datasets import load_digits

digits = load_digits() 
norm_data = digits.data / 16

model = np.load("model.npz")

l1_w = model["l1_w"]
l1_b = model["l1_b"]



l2_w = model["l2_w"]
l2_b = model["l2_b"]


l3_w = model["l3_w"]
l3_b = model["l3_b"]

l1 = Layer(l1_w, l1_b)
l2 = Layer(l2_w, l2_b)
l3 = Layer(l3_w, l3_b)


y1,y2,y3,probs = forward_pass(l1,l2,l3,norm_data)