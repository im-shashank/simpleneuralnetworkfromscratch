class optimize:

    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def gradient_decent_optimization(self, learning_rate):
        for p in self.parameters:
            p.data += -learning_rate * p.grad