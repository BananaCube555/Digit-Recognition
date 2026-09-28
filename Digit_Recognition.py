import numpy as np
from sklearn.datasets import load_digits
from backpropagation import backward_pass
from forward import relu, softMax, forward_pass, Layer

digits = load_digits() 
norm_data = digits.data / 16


w1 = np.random.rand(64, 32) * 0.01
w2 = np.random.rand(32, 16) * 0.01
w3 = np.random.rand(16, 10) * 0.01

b1 = np.zeros((1,32))
b2 = np.zeros((1,16))
b3 = np.zeros((1,10))

l1 = Layer(w1,b1)
l2 = Layer(w2,b2)
l3 = Layer(w3,b3)

y1,y2,y3,probs = forward_pass(l1,l2,l3,norm_data)


def loss_calculation(probs, ans):

    correct_class_preds = []

    for index in range(1797):
        current_image_probs = probs[index]
        pred_for_ans = current_image_probs[ans[index]]

        correct_class_preds.append(pred_for_ans)

    losses = -np.log(correct_class_preds)
    avr_loss = np.average(losses)
    return losses, avr_loss

all_answers = digits.target

losses, avr_loss = loss_calculation(probs, all_answers)

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

learning_rate = 0.1
iterations = 100

diff_losses = []

for i in range(iterations):

    y1, y2, y3, probs = forward_pass(l1, l2, l3, norm_data)

    new_loss, avr_new_loss = loss_calculation(probs, all_answers)
    diff_losses.append(avr_new_loss)

    dL_dW1, dL_dW2, dL_dW3 = backward_pass(
        y1, y2, probs, one_hot_answers, norm_data, l2.w, l3.w
    )

    l3.w -= learning_rate * dL_dW3
    l2.w -= learning_rate * dL_dW2
    l1.w -= learning_rate * dL_dW1

    if i % 1000 == 0:
        print("Iteration:", i, "Loss:", avr_new_loss)


# Run the trained network one final time
y1, y2, y3, probs = forward_pass(l1, l2, l3, norm_data)

_, final_loss = loss_calculation(probs, all_answers) # we only need the second return of the func

print("Final loss:", final_loss)

predictions = np.argmax(probs, axis=1)

accuracy = np.mean(predictions == all_answers)
print("Accuracy:", accuracy)

np.savez(
    "model.npz",
    l1_w=l1.w,
    l1_b=l1.b,
    l2_w=l2.w,
    l2_b=l2.b,
    l3_w=l3.w,
    l3_b=l3.b
)

