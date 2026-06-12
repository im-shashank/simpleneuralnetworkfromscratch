import numpy as nm

class value:

    def __init__(self, data, _childer()):
        self. data = data
        self.grad = 0.0
        self.backwards = lambda : None
        self.prev = set(_childer)

    def __add__(self, other): # self + other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data + other.data), (self, other))

        def backwards():
            self.grad += out.grad
            other.grad += out.grad
        out.backwards = self.backwards

        return out
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data * other.data) , (self, other))

        def backwards():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad
        out.backwards = backwards

        return out
    
    def __pow__(self, other): # self ** other
        other = other if isinstance(other, value) else value(other)
        out = value((self.data ** other.data), (self))

        def backwards():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
        out.backwards = backwards

        return out

    def __sub__(self, other): #self - other
        return self + (-other)
    
    def __truediv__(self, other): # self / other
        return (self * (other**-1))
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backwards()