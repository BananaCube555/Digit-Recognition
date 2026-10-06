import numpy as np
def backward_pass(y1, y2, probs, one_hot_answers, data, w2, w3,): 

    dL_dy3 = probs - one_hot_answers

    dL_dW3 = y2.T @ dL_dy3 / data.shape[0]
    dL_db3 = np.sum(dL_dy3, axis=0, keepdims=True) / data.shape[0]

    dL_dy2 = dL_dy3 @ w3.T
    dL_dz2 = dL_dy2 * (y2 > 0)

    dL_dW2 = y1.T @ dL_dz2 / data.shape[0]
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / data.shape[0]

    dL_dy1 = dL_dz2 @ w2.T
    dL_dz1 = dL_dy1 * (y1 > 0)

    dL_dW1 = data.T @ dL_dz1 / data.shape[0]
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / data.shape[0]

    return dL_dW1, dL_dW2, dL_dW3, dL_db1, dL_db2, dL_db3