if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(5)
x = np.random.rand(100,1)
y = 5 + 2 * x + np.random.rand(100,1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 1e-1
iters = 100

for i in range(iters):
    y_hat = predict(x)
    loss = F.mean_squared_error(y, y_hat)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data = W.data - lr * W.grad.data
    b.data = b.data - lr * b.grad.data

print(W, b)

import matplotlib.pyplot as plt

# numpy로 변환
x_np = x.data
y_np = y.data

# 예측값 (직선)
x_line = np.linspace(x_np.min(), x_np.max(), 100).reshape(-1, 1)
y_line = x_line @ W.data + b.data

# plot
plt.scatter(x_np, y_np, label='data')
plt.plot(x_line, y_line, label='prediction', linewidth=2, color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()