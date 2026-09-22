import numpy as np
from sklearn.datasets import load_digits


digits = load_digits() 
norm_data = digits.data / 16



class Layer:
    def __init__(self, w, b):
        self.w = np.random.rand(*w) * 0.01
        self.b = np.zeros((b)) 

    def forward(self, data):
         y = data @ self.w + self.b
         return y
    
def relu(y):
    y = np.maximum(0,y)
    return y  


def SoftMax(x):
    largestnum_x = np.max(x, axis=1, keepdims=True)
    x = x - largestnum_x
    x = np.exp(x)
    sums = np.sum(x, axis=1, keepdims=True)
    x = x / sums
        
    return x
    
# Runs the network

l1 = Layer((64,32), (1,32))
y1 = l1.forward(norm_data)

y1 = relu(y1)

l2 = Layer((32,16), (1,16))
y2 = l2.forward(y1)

y2 = relu(y2)

l3 = Layer((16,10), (1,10))
y3 = l3.forward(y2)

probs = SoftMax(y3)


def Loss_Calculation(probs, ans):

    correct_class_preds = []

    for index in range(1797):
        current_image_probs = probs[index]
        pred_for_ans = current_image_probs[ans[index]]

        correct_class_preds.append(pred_for_ans)

    losses = -np.log(correct_class_preds)
    avr_loss = np.average(losses)
    return losses, avr_loss

all_answers = digits.target

losses, avr_loss = Loss_Calculation(probs, all_answers)

print("Before:", avr_loss)


# makes the answers like this so we can calculate their gradient with probs
def one_hot_answers_func(answers):
    one_hot_answers = np.zeros((1797, 10))
    
    for index, value in enumerate(answers):
        one_hot_answer = np.zeros(10)
        one_hot_answer[value] = 1
        one_hot_answers[index] = one_hot_answer

    return one_hot_answers

        
one_hot_answers = one_hot_answers_func(all_answers)

#softmax-y3 = probs

w1 = l1.w
w2 = l2.w
w3 = l3.w

dL_dy3 = probs - one_hot_answers #get y3 gradients

# transpose y2 so its dimensions line up for the matrix multiplication
dL_dW3 = y2.T @ dL_dy3 / 1797

dL_dy2 = dL_dy3 @ w3.T

# (y2 > 0) is for the ReLU after l2 if the 
dL_dz2 = dL_dy2 * (y2 > 0)

dL_dW2 = y1.T @ dL_dz2 / 1797

dL_dy1 = dL_dz2 @ w2.T
dL_dz1 = dL_dy1 * (y1 > 0)
dL_dW1 = norm_data.T @ dL_dz1 / 1797


# devided with the gradient decides how big the change is 
learning_rate = 0.1

# Change the weights
# we use subtraction bc it works for both positive and negative numbers wthout needing extra checks
w3 -= learning_rate * dL_dW3
w2 -= learning_rate * dL_dW2
w1 -= learning_rate * dL_dW1


# Runs the network agian with the new weights
y1 = l1.forward(norm_data)
y1 = relu(y1)

y2 = l2.forward(y1)
y2 = relu(y2)

y3 = l3.forward(y2)

probs = SoftMax(y3)


losses, avr_loss = Loss_Calculation(probs, all_answers)

print("After", avr_loss)