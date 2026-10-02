import numpy as np



def loss_calculation(probs, ans):

    correct_class_preds = []

    for index in range(len(ans)):
        current_image_probs = probs[index]
        pred_for_ans = current_image_probs[ans[index]]

        correct_class_preds.append(pred_for_ans)

    losses = -np.log(correct_class_preds)
    avr_loss = np.average(losses)
    return losses, avr_loss







# makes the answers like this so we can calculate their gradient with probs
def one_hot_answers_func(answers):
    one_hot_answers = np.zeros((len(answers), 10))
    
    for index, value in enumerate(answers):
        one_hot_answer = np.zeros(10)
        one_hot_answer[value] = 1
        one_hot_answers[index] = one_hot_answer

    return one_hot_answers

        


