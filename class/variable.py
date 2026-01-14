import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y) # 객체 형태로 다시 돌려놓기
        return output
    
if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Function()
    y = f(x)
    print(y.data)