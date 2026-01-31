if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.))
y = x*x

y.backward(create_graph=True)
gx = x.grad # 변수(값)이 아닌 계산 그래프(식)
x.cleargrad()

z = gx ** 3 + y
z.backward()

print(x.grad)

