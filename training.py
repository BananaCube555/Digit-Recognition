import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from forward import relu, softMax, forward_pass, Layer
from Digit_Recognition import loss_calculation, one_hot_answers_func
from backpropagation import backward_pass

digits = load_digits() 

train_data, test_data, train_answers, test_answers = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

train_data = train_data / 16
test_data = test_data / 16



# Run the network once 
w1 = np.random.rand(64, 32) * 0.01
w2 = np.random.rand(32, 16) * 0.01
w3 = np.random.rand(16, 10) * 0.01

b1 = np.zeros((1,32))
b2 = np.zeros((1,16))
b3 = np.zeros((1,10))

model = np.load("model.npz")

l1_w = model["l1_w"]
l1_b = model["l1_b"]

l2_w = model["l2_w"]
l2_b = model["l2_b"]

l3_w = model["l3_w"]
l3_b = model["l3_b"]

l1 = Layer(l1_w, l1_b)
l2 = Layer(l2_w,l2_b)
l3 = Layer(l3_w,l3_b)

learning_rate = 0.023

one_hot_answers = one_hot_answers_func(train_answers)

class Training:
    def __init__(self,learning_rate,l1,l2,l3,training_data,training_answers,one_hot_answers):
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.training_answers = training_answers
        self.one_hot_answers = one_hot_answers
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        

    def train(self, iterations):
        for i in range(iterations):

            y1,y2,y3,probs = forward_pass(self.l1,self.l2,self.l3,self.training_data)

            losses, avr_losss = loss_calculation(probs,self.training_answers)

            dL_dW1, dL_dW2, dL_dW3 = backward_pass(y1,y2,probs,self.one_hot_answers,self.training_data, self.l2.w, self.l3.w)

            # Update the weights
            self.l3.w -= self.learning_rate * dL_dW3
            self.l2.w -= self.learning_rate * dL_dW2
            self.l1.w -= self.learning_rate * dL_dW1

            if i % 500 == 0:
                            print("Iteration:", i, "Loss:", avr_losss)
        return self.l1.w, self.l2.w, self.l3.w


tinput = int(input("Train ? (1 YES, 0 NO) "))

if tinput == 1:

    trainer = Training(
        learning_rate,
        l1,
        l2,
        l3,
        train_data,
        train_answers,
        one_hot_answers
    )

    l1_w, l2_w, l3_w = trainer.train(25000)

    l1 = Layer(l1_w, l1_b)
    l2 = Layer(l2_w, l2_b)
    l3 = Layer(l3_w, l3_b)

    np.savez(
        "model.npz",
        l1_w=l1_w,
        l1_b=l1_b,
        l2_w=l2_w,
        l2_b=l2_b,
        l3_w=l3_w,
        l3_b=l3_b
    )

    

T_y1, T_y2, T_y3, T_probs = forward_pass(l1, l2, l3, test_data)

predictions = np.argmax(T_probs, axis=1)

correct = (predictions == test_answers) #[True, False, False, ....]

accuracy = np.mean(correct) #calc the avr of 0,1,0,0 True false ..

print("Accuracy:", accuracy * 100, "%")

