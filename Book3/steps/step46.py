if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.models import MLP
from dezero.optimizers import SGD, MomentumSGD
import dezero.functions as F
import dezero.layers as L


np.random.seed(5)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

H1, H2, O = 8, 8, 1

lr = 1e-1
iters = 10000

model = MLP((H1, H2, O))
#optimizer = SGD(lr)
optimizer = MomentumSGD(lr)
optimizer.setup(model)


for i in range(iters):
    y_hat = model(x)
    loss = F.mean_squared_error(y, y_hat)

    model.cleargrads()

    loss.backward()

    optimizer.update()

    if i % 2000 == 0:
        print(loss)


import matplotlib.pyplot as plt

# numpy로 변환 (x, y는 이미 numpy 배열)
x_np = x
y_np = y

# 예측값 (NN 전체 통과)
x_line = np.linspace(x_np.min(), x_np.max(), 200).reshape(-1, 1)
x_line_var = Variable(x_line)
y_line = model(x_line_var).data

# plot
plt.scatter(x_np, y_np, label='data', s=10)
plt.plot(x_line, y_line, label='prediction', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()