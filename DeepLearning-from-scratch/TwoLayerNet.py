#
# 第四章 勾配法の実装
#

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist 
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    #
    # 損失関数
    #
    def loss(self, y, ans):
        # 平均二乗法
        #p = np.sum((y - ans) ** 2) / 2
        #return p

        # クロスエントロピー誤差
        if y.ndim == 1:
            ans = ans.reshape(1, ans.size)
            y = y.reshape(1, y.size)
        # 教師データがone-hot表現の場合、正解ラベルのインデックスに変換
        if ans.size == y.size:
            ans = ans.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), ans] + 1e-7)) / batch_size
    
    def numerical_gradient(self, x, ans):
        loss_W = lambda dummy: self.loss(self.predict(x), ans)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
            
        batch_num = x.shape[0]
            
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
            
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)
    return x_train, t_train

#
#
#

def train():
    x_train, t_train = get_data()
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    loss_results = []

    for i in range(10000):
        # 訓練データをランダムに取り出す
        batch_mask = np.random.choice(len(x_train), size=1000, replace=False)
        x = x_train[batch_mask]
        ans = t_train[batch_mask]

        z2 = net.predict(x)
        y = net.loss(z2, ans)
        loss_results.append(y)
        # print(str(i) + " 回目のz2: " + str(z2[0:10]))
        print(str(i) + " 回目の損失関数: " + str(y))
        grad = net.gradient(x, ans)
        net.params['W1'] -= 0.05 * grad['W1']
        net.params['b1'] -= 0.05 * grad['b1']
        net.params['W2'] -= 0.05 * grad['W2']
        net.params['b2'] -= 0.05 * grad['b2']

        # 50回ごとにネットワークを保存
        # if i % 50 == 0:
        #     with open(f'trained_params_{i}.pkl', 'wb') as f:
        #        pickle.dump(net.params, f)

    # グラフの描画
    plt.plot(loss_results)

    plt.xlabel("index")
    plt.ylabel("the number of loss function")
    plt.title("Training loss curve")
    plt.grid()
    plt.show()

    # ネットワークの保存
    with open('trained_params.pkl', 'wb') as f:
        pickle.dump(net.params, f)

def test():
    # 検証回数
    num_tests = 100

    # ネットワークの読み込み
    with open('trained_params.pkl', 'rb') as f:
        params = pickle.load(f)

    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    net.params = params

    x_train, t_train = get_data()
    batch_mask = np.random.choice(len(x_train), size=num_tests, replace=False)
    x = x_train[batch_mask]
    ans = t_train[batch_mask]

    # 計算
    z2 = net.predict(x)
    y = net.loss(z2, ans)

    # 合ってるかどうか検証
    p = np.argmax(z2, axis=1)
    t = np.argmax(ans, axis=1)

    print("回答: ", p)
    print("解答: ", t)
    accuracy = np.sum(p == t) / len(t)
    print("正解数: ", np.sum(p == t))
    print("検証数: ", len(t))
    print("正解率: " + str(accuracy * 100) + "%")

    # 画像の表示
    img = (x * 256).reshape(28 * num_tests, 28)
    show_image(img)

if __name__ == "__main__":
    train()
    test()