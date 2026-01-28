if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F

max_epoch =100
batch_size =30
hidden_size = 8
lr = 1e-2

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size)

loss_list = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i*batch_size:(i+1)*batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    loss_list.append(avg_loss)
    print('epoch %d, loss %.2f'%(epoch+1, avg_loss))

# --- loss 그래프 ---
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, max_epoch + 1), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training Loss')
plt.show()

# --- 결정 경계 ---
x = np.array([example[0] for example in train_set])
t = np.array([example[1] for example in train_set])

h = 0.01
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    scores = model(grid)
pred = np.argmax(scores.data, axis=1).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, pred, cmap=plt.cm.Spectral, alpha=0.7)
plt.scatter(x[:, 0], x[:, 1], c=t, cmap=plt.cm.Spectral, s=10)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Decision Boundary')
plt.show()