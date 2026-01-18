import numpy as np
import weakref # 약한 참조: 참조 카운터를 증가시키지 않음 (순환 참조 해결)
from config import Config
import contextlib

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# with 구문에서 이용 시 전처리, 후처리
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # config에서 기본 값 (True) 가져오기
    setattr(Config, name, value) # config 값 주어진 값으로 설정 (전처리)
    try:
        yield
    finally:
        setattr(Config, name, old_value) # config 값 돌려놓기 (후처리)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None 
        self.generation = 0


    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set() # 중복 제거

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx 

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # weak reference 참조 : y()


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            
            for output in outputs:
                output.set_creator(self)
        
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0] 
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

if __name__ == "__main__":
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x))) 
    y.backward()

    Config.enable_backprop = False
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x))) 

    with no_grad():
        x = Variable(np.array(2.))
        y = square(x)
        print(y.data) 



