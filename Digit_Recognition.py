import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import math

digits = load_digits() 
norm_data = digits.data / 16



class Layer:
    def __init__(self, w, b, ):
        self.w = np.random.rand(*w) * 0.01 # *w to unload the tuple, input needs 2 numbers
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

l1 = Layer((64,32), (1,32))
y1 = l1.forward(norm_data)

y1 = relu(y1)

l2 = Layer((32,16), (1,16))
y2 = l2.forward(y1)

y2 = relu(y2)

l3 = Layer((16,10), (1,10))
y3 = l3.forward(y2)

probs = SoftMax(y3)

# =========================
# VISUALIZATION 
# =========================

def create_prediction(index):
    prediction_probs = probs[index]
    prediction = np.argmax(prediction_probs) # Returns the index of the largest probability
    answer = digits.target[index] # Actual digit label for image x

    return prediction_probs, prediction, answer

# -- DISPLAY IMAGE --
def display_image(index, prediction, answer):
    plt.imshow(digits.images[index], cmap='gray')
    plt.title(f"Prediction: {prediction} Actual: {answer}")
    plt.show()


#  -- DISPLAY PREDICTIONS --
def display_predictions(prediction, predictions_probs, answer):

    categories = np.arange(0, 10) # Generates numbers from 0 to 9

    plt.bar(categories , predictions_probs)
    plt.title(f"Prediction: {prediction} Actual: {answer}")

    plt.show()


# index = int(input("index: "))
# prediction_probs, prediction, answer = create_prediction(index)
# display_image(index, prediction, answer)
# display_predictions(prediction, prediction_probs, answer)
# -- BROKEN LOGIG FOR LOSS CALC FUNCTION--
def Loss_Calculation(probs, ans):
    
    correct_class_preds = []

    for index in range(1797): # TODO: Implement a non loop way 
        current_image_probs = probs[index]
        pred_for_ans = current_image_probs[ans[index]]

        correct_class_preds.append(pred_for_ans)

    losses = -np.log(correct_class_preds)
    return losses


all_answers = digits.target

Loss_Calculation(probs, all_answers)
