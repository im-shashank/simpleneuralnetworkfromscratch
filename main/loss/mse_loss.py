class mse:
    
    def __init__(self, ys, ypred):
        self.ys = ys
        self.ypred = ypred

    def calculate_loss(self):
        loss = sum((yout - ygt)**2 for ygt, yout in zip(self.ys, self.ypred))
        return loss