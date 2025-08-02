#
# 第三章の実装
#

import numpy as np
from dataset.mnist import load_mnist 
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=False, one_hot_label=True)
    return x_test, t_test

def init_network():
     with open("deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network
    
"""
    network = {}
    network['W1'] = np.array([0.2, 0.2, 0.2])
    network['W2'] = np.array([0.1, 0.2, 0.3])
    network['W3'] = np.array([0.4, 0.5, 0.6])
    network['b1'] = np.array([0.1, 0.1, 0.1])
    network['b2'] = np.array([0.1, 0.1, 0.1])
    network['b3'] = np.array([0.1, 0.1, 0.1])
"""

def predict(x):
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    z1 = sigmoid(np.dot(x, W1) + b1)
    z2 = sigmoid(np.dot(z1, W2) + b2)
    z3 = softmax(np.dot(z2, W3) + b3)
    return z3



get_data()
id = 5
x_test, t_test = get_data()
print(t_test[id])
x = x_test[id]

output = predict(x)
print(output * 100)  # 出力をパーセント表示
print(np.argmax(output))  # 最大値のインデックスを取得



