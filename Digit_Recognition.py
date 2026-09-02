import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import math

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
    prediction = np.argmax(prediction_probs)
    answer = digits.target[index]

    return prediction_probs, prediction, answer

# -- DISPLAY IMAGE --
def display_image(index, prediction, answer):
    plt.imshow(digits.images[index], cmap='gray')
    plt.title(f"Prediction: {prediction} Actual: {answer}")
    plt.show()


#  -- DISPLAY PREDICTIONS --
def display_predictions(prediction, predictions_probs, answer):

    categories = np.arange(0, 10)

    plt.bar(categories , predictions_probs)
    plt.title(f"Prediction: {prediction} Actual: {answer}")

    plt.show()


# index = int(input("index: "))
# prediction_probs, prediction, answer = create_prediction(index)
# display_image(index, prediction, answer)
# display_predictions(prediction, prediction_probs, answer)

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

losses, old_loss = Loss_Calculation(probs, all_answers)

# -- GRADIENT --


def forward_Calc(data, layers):
    
        y1 = relu(layers[0].forward(data))
        y2 = relu(layers[1].forward(y1))
        y3 = (layers[2].forward(y2))

        return y1, y2 , y3

# Proggress of remake of first version of calc_gradient

# layer_name = l3, change_of_weight = 0.0001, old_loss = old_loss, layer_names = [l1,l2,l3], answer = digits.target or all_answers

def calculate_gradient(layer_name, row, col, change_of_weight, old_loss, data, layer_names, answers): 
    layer_name.w[row][col] += change_of_weight

    y1, y2, y3 = forward_Calc(data,layer_names)
    probs = SoftMax(y3)

    new_losses, new_avr_loss = Loss_Calculation(probs, answers)

    new_gradient = (new_avr_loss - old_loss) / change_of_weight

    layer_name.w[row][col] -= change_of_weight

    return new_gradient


def calculate_gradients(layer):
    
    gradients = np.zeros_like(layer.w) #Creates a matrix of zeros in the shape of layer.w

    
    for row in range(layer.w.shape[0]): # for row in range(16)
        for col in range(layer.w.shape[1]): # for col in range(10)
            gradients[row][col] = calculate_gradient(layer, row, col, change_of_weight=0.0001, old_loss=old_loss, data=norm_data ,layer_names=[l1,l2,l3], answers=digits.target ) 

    return gradients


    

# --TRAINING--

learning_rate = 0.1

# avr_loss, 

for step in range(10):
    
    
    l1_gradients = calculate_gradients(l1)
    l2_gradients = calculate_gradients(l2)
    l3_gradients = calculate_gradients(l3)

    l1.w -= learning_rate * l1_gradients
    l2.w -= learning_rate * l2_gradients
    l3.w -= learning_rate * l3_gradients

    # Runs the layers and calculates the loss
    y1, y2, y3 = forward_Calc(norm_data, [l1,l2,l3])
    y3 = SoftMax(y3)
    
    new_losses, new_avr_loss = Loss_Calculation(y3, all_answers)

    print(step, new_avr_loss) # prints current number of loop and the loss

    old_loss = new_avr_loss # The loss that got calculated is now old or current loss for the next iteration


print("test")