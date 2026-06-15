class optimize:

    def __init__(self, parameters):
        self.parameters = parameters

    def gradient_decent_optimization(self, learning_rate):
        for p in self.parameters:
            p.data += -learning_rate * p.grad