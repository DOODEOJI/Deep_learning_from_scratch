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
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # list unpacking

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
    
        self.inputs = inputs
        self.outputs = outputs
        
        return outputs if len(outputs) > 1 else outputs[0] # 원소 하나라면 튜플이 아닌 해당 객체만 반환
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy # add의 미분값은 1이므로 미분값을 그대로 흘려보냄
    
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
    y = Variable(np.array(3.))

    z = add(square(x), square(y))
    z.backward()

    print(z.data)
    print(x.grad)
    print(y.grad)

