if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
from dezero import DataLoader

max_epoch = 5
batch_size = 100
hidden_size = 1000
lr = 1.0

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

model = MLP((hidden_size, hidden_size, hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss_list.append(sum_loss / len(train_set))
    train_acc_list.append(sum_acc / len(train_set))
    print("epoch: {}".format(epoch+1))
    print("train loss : {:.4f}, accuracy : {:.4f}".format(train_loss_list[-1], train_acc_list[-1]))

    sum_loss, sum_acc = 0, 0

    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss_list.append(sum_loss / len(test_set))
    test_acc_list.append(sum_acc / len(test_set))
    print("test loss : {:.4f}, accuracy : {:.4f}".format(test_loss_list[-1], test_acc_list[-1]))

# --- loss 그래프 ---
import matplotlib.pyplot as plt

epochs = range(1, max_epoch + 1)

plt.figure()
plt.plot(epochs, train_loss_list, label='train')
plt.plot(epochs, test_loss_list, label='test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.show()

# --- accuracy 그래프 ---
plt.figure()
plt.plot(epochs, train_acc_list, label='train')
plt.plot(epochs, test_acc_list, label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()