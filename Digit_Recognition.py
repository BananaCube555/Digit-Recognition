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

# l1 = Layer((64,32), (1,32))
# y1 = l1.forward(norm_data)

# y1 = relu(y1)

# l2 = Layer((32,16), (1,16))
# y2 = l2.forward(y1)

# y2 = relu(y2)

# l3 = Layer((16,10), (1,10))
# y3 = l3.forward(y2)

# probs = SoftMax(y3)


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


def one_hot_vector_answers(answers):
    one_hot_answers = np.zeros((1797, 10))
    
    for i in answers:
        pass

