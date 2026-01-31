import numpy as np

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
            x, y = f.input, f.output # 반복문이므로 self.grad로 넣어주면 계속 y.grad가 입력되므로
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        
        output.set_creator(self)
        self.input = input # 입력 변수를 보관해야 도함수에 대입해서 미분 값을 구할 수 있음
        self.output = output
        
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy 
        # gradient를 역방향으로 누적
        # dy/dy (gy) * 2x (제곱 함수 미분 시 2x에 저장한 함수 입력값 대입) = 다음 backward로 보낼 gy (누적된 가중치- dy/dy * dy/db)가 됨
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x): # 0차원 ndarray에 대한 연산 시 결과가 scalar로 반환됨 (float, int 등) 따라서 변환 필요
    if np.isscalar(x):
        return np.array(x)
    return x
    
 
if __name__ == "__main__":

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()

    print(x.grad)