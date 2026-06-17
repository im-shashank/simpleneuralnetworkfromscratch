import math

class value:

    def __init__(self, data, _childern=()):
        self. data = data
        self.grad = 0.0
        self.backwards = lambda : None
        self.prev = set(_childern)

    def __add__(self, other): # self + other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data + other.data), (self, other))

        def backwards():
            self.grad += out.grad
            other.grad += out.grad
        out.backwards = backwards

        return out
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data * other.data) , (self, other))

        def backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backwards = backwards

        return out
    
    def __pow__(self, other): # self ** other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data ** other.data), (self,))

        def backwards():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
        out.backwards = backwards

        return out

    def __sub__(self, other): #self - other
        return self + (-other)
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def relu(self):
        out = value((0 if self.data <= 0 else self.data), (self,))

        def backwards():
            self.grad += (out.data > 0) * out.grad
        out.backwards = backwards

        return out
    
    def tanh(self):
        x = self.data
        # t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        t = math.tanh(x)
        out = value (t, (self,))

        def backwards():
            self.grad += (1 - t**2) * out.grad
        out.backwards = backwards

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v.backwards()