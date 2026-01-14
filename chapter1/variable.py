import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

# square 함수로 고정된 function
'''
class Function:
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y) # 객체 형태로 다시 돌려놓기
        return output
'''

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, input):
        raise NotImplementedError()
    
class Square(Function): # Function class 상속
    def forward(self, x):
        return x ** 2
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))    
    
def numerical_diff(f, x, eps=1e-4): # 중앙차분(centered difference)을 이용한 수치미분(numerical differentiation)
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2*eps)



 
if __name__ == "__main__":
    '''
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(y.data)

    # 합성함수 계산
    x = Variable(np.array(0.5))
    A = Square()
    B = Exp()
    C = Square()

    a = A(x)
    b = B(a)
    y = A(b) # C 대신 A 활용해도 됨

    print(y.data)

    # 수치미분 계산
    f = Square()
    x = Variable(np.array(2.))
    dy = numerical_diff(f, x)
    print(dy)
    '''

    # 합성함수 수치미분
    x = Variable(np.array(0.5))
    dy = numerical_diff(f,x) # f 함수 선언하고 함수 객체로 전달 (인수 없어도 됨)
    print(dy)