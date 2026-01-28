if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.models import MLP
from dezero.optimizers import SGD, MomentumSGD
import dezero.functions as F
import dezero.layers as L

model = MLP((10,3))

x = Variable(np.array([[0.2, -0.4],[0.5,0.5],[1.3,-3.2],[2.1,0.3]]))
t = np.array([2,0,1,0]) #정답 레이블
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)

print(loss)