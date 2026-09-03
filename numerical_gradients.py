# I made these functions so i could understand how gradients work so they are not reusable and they are slow

# layer_name = l3, change_of_weight = 0.0001, old_loss = old_loss, layer_names = [l1,l2,l3], answer = digits.target or all_answers

# def calculate_gradient(layer_name, row, col, change_of_weight, old_loss, data, layer_names, answers): 
#     layer_name.w[row][col] += change_of_weight

#     y1, y2, y3 = forward_Calc(data,layer_names)
#     probs = SoftMax(y3)

#     new_losses, new_avr_loss = Loss_Calculation(probs, answers)

#     new_gradient = (new_avr_loss - old_loss) / change_of_weight

#     layer_name.w[row][col] -= change_of_weight

#     return new_gradient


# def calculate_gradients(layer):
    
#     gradients = np.zeros_like(layer.w) #Creates a matrix of zeros in the shape of layer.w

    
#     for row in range(layer.w.shape[0]): # for row in range(16)
#         for col in range(layer.w.shape[1]): # for col in range(10)
#             gradients[row][col] = calculate_gradient(layer, row, col, change_of_weight=0.0001, old_loss=old_loss, data=norm_data ,layer_names=[l1,l2,l3], answers=digits.target ) 

#     return gradients

