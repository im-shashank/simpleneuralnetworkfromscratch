import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.MLP.mlp import *
from main.loss.mse_loss import *
from main.optimize.optimize import *

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] # desired output

x = [2.0, 3.0, -1.0]
m = mlp(len(x), [4, 4, 1])

learning_rate = 0.001
acceptable_loss = 0.00001

print(f"\nparameters before training: \n {m.parameters()}\n\n")

while True:
    ypred = [m(x) for x in xs]
    # print(f"\n\n predictions are :\n {[d.data for d in ypred]}\n")

    mse_loss = mse(ys, ypred)
    loss = mse_loss.calculate_loss()
    loss.backward()

    print(f"current loss is {loss.data}")
    if loss.data <= acceptable_loss:
        print(f"\n\n final predictions are :\n {[d.data for d in ypred]}\n")
        print(f"\n\nthe trained parameters are: \n {m.parameters()}")
        break

    optimization = optimize(m.parameters())
    optimization.gradient_decent_optimization(learning_rate)