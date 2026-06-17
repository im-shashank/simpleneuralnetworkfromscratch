#### Disclaimer: ####

I am creating this project as an exercise on my way to learn neural networks, deep learning, etc.

This project contains a simple neural network (to be created) and a simple implementation of backwards and forwards propagation in a neural network.
Right now the backwards and forwards propagation is very simple, containing the basic mathematical operations of addition, multiplication, subtraction,
division and power.

I hope to increase the number of mathematical operations in future.

This project is not being created as a replacement or competition to pytorch and tensor flow. It is just for me to practice. 

Use pytorch or tensor flow for serious work.

---------------------------------------------------------------------------

#### What I aim to achieve with this project: ####

This project is my implementation of a small part of pytorch library or my take on micrograd library by Andrej Karpathy as I am learning neural networks from him.

With this project I am able to train a simple 41 parameter neural network to predict numbers accurately based on some test data.

You can try running test-predictions.py file to checkout the results for yourself.

---------------------------------------------------------------------------

#### How I wish to expand upon this project: ####

In future I wish to add more activation functions. Right now we have only tanh as activation function.

In future I might mould this project to predict some more things apart from just numbers. But that'll have to wait.

---------------------------------------------------------------------------

#### What you can do with this project: ####

Clone it, modify it.

Thanks for checking out my project.

---------------------------------------------------------------------------

#### How to use it: ####

Initial the multi-layered perceptron (mlp) by giving number of input and the neural archtecture (number of layers and number of neurons per layer)

Pass the test data set. See the examples in test folder for more details.

Start your training loop.

See the final predictions.

For better understanding look into the test folders for usage examples.

---------------------------------------------------------------------------

#### What tests I have ran on this auto grad library ####

A forward pass test

A simple prediction test

A quadrictic equation prediction test

```text
Iteration 9200 | Loss: 0.002186160534238945
Iteration 9300 | Loss: 0.0021746214919922597
Iteration 9400 | Loss: 0.0021630348760088443
Iteration 9500 | Loss: 0.0021517049148426755
Iteration 9600 | Loss: 0.002140601880534238
Iteration 9700 | Loss: 0.002129707439662365
Iteration 9800 | Loss: 0.002119039218067881
Iteration 9900 | Loss: 0.0021085482585661017
Iteration 10000 | Loss: 0.002098247116354708

--- Results ---
Prediction for x=20: 399.60 (Expected: 400.00)
```
