from chapter1.improvement_ab import Variable, Function, Square, Exp, square, exp
from chapter1.variable import numerical_diff
import unittest
import numpy as np

class SquareTest(unittest.TestCase):
    
    def test_forward(self): # 테스트 시에는 test + a 메세드 만들면 됨
        x = Variable(np.array(2.))
        y = square(x)
        expected = np.array(4.)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

if __name__ == '__main__':
    unittest.main(verbosity=2)