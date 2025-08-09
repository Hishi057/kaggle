#
# 第四章 勾配法の実装
#

import numpy as np
from dataset.mnist import load_mnist 
from PIL import Image
import pickle


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(784, 10) * 0.1
        self.b = 0

    def predict(self, x):
        result = sigmoid(np.dot(x, self.W) + self.b)
        return result

    def loss(self, x, t):
        p = np.sum((x - t) ** 2) / 2 #平均二乗法
        return p

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
        normalize=True, one_hot_label=True)
    return x_train, t_train

x_train, t_train = get_data()
net = simpleNet()

def numerical_gradient(f, x):
    grad = np.zeros_like(x)
    h = 1e-4

    for idx in (range(x.size)):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x) 
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
    return grad

def f(dummy):
    z1 = net.predict(x)
    y = net.loss(z1, t_train[0:x.shape[0]])
    return y

batch_size = 10
x = x_train[0:1][0]

# 何回勾配を計算するか？
num_iterations = 1000
for i in range(num_iterations):
    grad = numerical_gradient(f, W)
    print(str(i) + "回目の損失関数: "+ str(f(x)))
    # xを更新
    x -= 1e-4 * grad