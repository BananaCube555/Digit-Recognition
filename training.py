import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from forward import relu, softMax, forward_pass, Layer
from Digit_Recognition import loss_calculation,one_hot_answers_func
from backpropagation import backward_pass

digits = load_digits() 

train_data, test_data, train_answers, test_answers = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

train_data = train_data / 16
train_answers = train_answers 
test_data = test_data / 16
test_answers = test_answers 


# Run the network once 
w1 = np.random.rand(64, 32) * 0.01
w2 = np.random.rand(32, 16) * 0.01
w3 = np.random.rand(16, 10) * 0.01

b1 = np.zeros((1,32))
b2 = np.zeros((1,16))
b3 = np.zeros((1,10))


l1 = Layer(w1, b1)
l2 = Layer(w2,b2)
l3 = Layer(w3,b3)

learning_rate = 0.001

one_hot_answers = one_hot_answers_func(train_answers)

class training:
    def __init__(self):
        pass

    def train(self,learning_rate):
        y1,y2,y3,probs = forward_pass(l1,l2,l3,train_data)
        losses, avr_losss = loss_calculation(probs,train_answers)
        dL_dW1, dL_dW2, dL_dW3 = backward_pass(y1, y2, probs, one_hot_answers, train_data, l2.w, l3.w)

        # Update the weights
        l3.w -= learning_rate * dL_dW3
        l2.w -= learning_rate * dL_dW2
        l1.w -= learning_rate * dL_dW1

