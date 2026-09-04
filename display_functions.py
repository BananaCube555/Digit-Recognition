# =========================
# VISUALIZATION 
# =========================

# def create_prediction(index, probs):
#     prediction_probs = probs[index]
#     prediction = np.argmax(prediction_probs)
#     answer = digits.target[index]

#     return prediction_probs, prediction, answer

# -- DISPLAY IMAGE --
# def display_image(index, prediction, answer):
#     plt.imshow(digits.images[index], cmap='gray')
#     plt.title(f"Prediction: {prediction} Actual: {answer}")
#     plt.show()


 #  -- DISPLAY PREDICTIONS --
# def display_predictions(prediction, predictions_probs, answer):

#     categories = np.arange(0, 10)

#     plt.bar(categories , predictions_probs)
#     plt.title(f"Prediction: {prediction} Actual: {answer}")

#     plt.show()


# index = int(input("index: "))
# prediction_probs, prediction, answer = create_prediction(index)
# display_image(index, prediction, answer)
# display_predictions(prediction, prediction_probs, answer)
