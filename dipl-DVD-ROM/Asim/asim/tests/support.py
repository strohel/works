from unittest import TestCase


class F(Function):
    def eval(self, x):
        return 3/(x+1) + 5/(x+2) + 18/(x+3) + 1/x - 3.

class G(Function):
    def eval(self, x):
        return -1/(x**2) -3/(x+1)**2 -5/(x+2)**2 -18/(x+3)**2

class TestSupport(TestCase):

    def test_newton(self):
        self.assertAlmostEqual(find_root(F(), G(), 1., 1e-8), 6.60784373 )
