import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None 

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None: # gradient 누적해주기 (같은 변수 사용 시 누적되도록)
                    x.grad = gx
                else:
                    x.grad = x.grad + gx 
                    # 복합 대입 연산자 (Inplace) 사용 시 y.grad를 x.grad에 누적 시킬 때 
                    # y.grad를 복사하는 것이 아닌 동일한 ID에 누적하므로 y.grad 값이 오염됨

                if x.creator is not None:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
    
        self.inputs = inputs
        self.outputs = outputs
        
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
    x = Variable(np.array(2.))
    y = add(x, x) 
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = add(add(x,x),x)
    y.backward()
    print(x.grad)

