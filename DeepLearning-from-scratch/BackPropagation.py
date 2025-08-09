import numpy as np
from dataset.mnist import load_mnist 
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from collections import OrderedDict

def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)
    return x_train, t_train

def softmax(x):
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x

#
# 損失関数
#
def cross_entropy_error(y, ans):
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

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        self.dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return self.dx

class SoftmaxWithLoss:
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.x = x
        self.t = t
        self.y = softmax(x)

        # 損失関数で評価
        # out = np.sum((self.y ** 2)/2)
        out = cross_entropy_error(self.y, t)
        return out

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class Network:
    def __init__(self):
        self.params = {}
        self.params['W1'] = np.random.randn(784, 144) * 0.1
        self.params['b1'] = np.zeros(144)
        self.params['W2'] = np.random.randn(144, 64) * 0.1
        self.params['b2'] = np.zeros(64)
        self.params['W3'] = np.random.randn(64, 25) * 0.1
        self.params['b3'] = np.zeros(25)
        self.params['W4'] = np.random.randn(25, 10) * 0.1
        self.params['b4'] = np.zeros(10)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Sigmoid3'] = Sigmoid()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.lastlayer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        L = self.lastlayer.forward(y, t)
        return L
    
    def accuracy(self, x, t):
        y = self.predict(x)
        print(np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)))
        accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / float(x.shape[0])
        return accuracy


    def backPropagetion(self, x, t):
        L = self.loss(x, t)
        dout = 1
        dout = self.lastlayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db

        return grads

x_train, t_train = get_data()
net = Network()


loss_list = []

# 何回勾配を計算するか？
num_iterations = 10001
for i in range(num_iterations):
    batch_mask = np.random.choice(len(x_train), size=100, replace=False)
    x = x_train[batch_mask]
    ans = t_train[batch_mask]
    
    L = net.loss(x, ans)
    loss_list.append(L)
    print(str(i) + "回目の損失関数: " + str(L))

    grads = net.backPropagetion(x, ans)

    learning_rate_max = 1
    learning_rate_min = 0.0001

    learning_rate = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * (1 + np.cos(i / num_iterations* np.pi))


    # net.params のキーをループ
    for key in net.params.keys():
        net.params[key] -= learning_rate * grads[key]

print("1回目の損失関数: " + str(loss_list[1]))
print("1000回目の損失関数: " + str(loss_list[1000]))
print("5000回目の損失関数: " + str(loss_list[5000]))
print("10000回目の損失関数: " + str(loss_list[10000]))

# グラフの描画
plt.plot(loss_list)

plt.xlabel("index")
plt.ylabel("the number of loss function")
plt.title("Training loss curve")
plt.grid()
plt.show()

# ネットワークの保存
with open('trained_params.pkl', 'wb') as f:
    pickle.dump(net.params, f)


# 検証回数
num_tests = 1000

# ネットワークの読み込み
with open('trained_params.pkl', 'rb') as f:
    params = pickle.load(f)

net.params = params

x_test, t_test = get_data()
batch_mask = np.random.choice(len(x_test), size=num_tests, replace=False)
x = x_train[batch_mask]
t = t_train[batch_mask]

print("検証数: " + str(num_tests) + " 回")
print("正解率: " + str(net.accuracy(x, t) * 100) + " %")

# 画像の表示
img = (x * 256).reshape(28 * num_tests, 28)
show_image(img)