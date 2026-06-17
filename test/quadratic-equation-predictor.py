import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.MLP.mlp import *
from main.loss.mse_loss import *
from main.optimize.optimize import *
import math

# traning dataset
xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ys = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]

# Proper Z-Score Normalization (Crucial for learning speed)
mean_x = sum(xs) / len(xs)
std_x = (sum((x - mean_x)**2 for x in xs) / len(xs))**0.5

mean_y = sum(ys) / len(ys)
std_y = (sum((y - mean_y)**2 for y in ys) / len(ys))**0.5

xs_norm = [(x - mean_x) / std_x for x in xs]
ys_norm = [(y - mean_y) / std_y for y in ys]

m = mlp(1, [32,32, 1])

learning_rate = 0.0001
acceptable_loss = 0.001

max_iteration = 10000
iteration = 0

loss_prev = 0

optimization = optimize(m.parameters())

while True and iteration <= max_iteration:
    # print(iteration)

    optimization.zero_grad()

    # Forward pass with normalized X
    ypred = [m([x]) for x in xs_norm]
    # print(ypred)

    # Calculate MSE with normalized Y
    mse_loss = mse(ys_norm, ypred)
    loss = mse_loss.calculate_loss()
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration} | Loss: {loss.data}")

    loss.backward()

    if loss.data <= acceptable_loss:
        break

    optimization.gradient_decent_optimization(learning_rate)

    iteration += 1

# Inference / Testing
x_input = 20

# Scale the input using the exact same logic we used for training
x_input_norm = (x_input - mean_x) / std_x

# Get the normalized prediction
y_output_norm = m([x_input_norm])

# Reverse the scaling to get the real-world value
y_output_real = (y_output_norm.data * std_y) + mean_y

print("\n--- Results ---")
print(f"Prediction for x={x_input}: {y_output_real:.2f} (Expected: 400.00)")