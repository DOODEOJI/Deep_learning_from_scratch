if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(5)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)
x, y = Variable(x), Variable(y)

I, H1, H2, O = 1, 8, 8, 1

W1 = Variable(0.01 * np.random.randn(I,H1))
b1 = Variable(np.zeros(H1))

W2 = Variable(0.01 * np.random.randn(H1,H2))
b2 = Variable(np.zeros(H2))

W3 = Variable(0.01 * np.random.randn(H2,O))
b3 = Variable(np.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.relu(y)
    y = F.linear(y, W2, b2)
    y = F.relu(y)
    y = F.linear(y, W3, b3)
    return y

lr = 1e-1
iters = 10000

for i in range(iters):
    y_hat = predict(x)
    loss = F.mean_squared_error(y, y_hat)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    W3.cleargrad()
    b3.cleargrad()

    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    W3.data -= lr * W3.grad.data
    b3.data -= lr * b3.grad.data

    if i % 2000 == 0:
        print(loss)


import matplotlib.pyplot as plt

# numpy로 변환
x_np = x.data
y_np = y.data

# 예측값 (NN 전체 통과)
x_line = np.linspace(x_np.min(), x_np.max(), 200).reshape(-1, 1)
x_line_var = Variable(x_line)
y_line = predict(x_line_var).data

# plot
plt.scatter(x_np, y_np, label='data', s=10)
plt.plot(x_line, y_line, label='prediction', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()