if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F

class TwoLayerModel(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.L1 = L.Linear(hidden_size)
        self.L2 = L.Linear(out_size)

    def forward(self, x):
        y = self.L1(x)
        y = F.relu(y)
        y = self.L2(y)
        return y
    
x = Variable(np.random.randn(5, 10), name ='x')
model = TwoLayerModel(100, 10)
model.plot(x)