import numpy as nm
from ..value.value import *

class neuron:

    def __init__(self, data):
        self.weight = nm.random(0,1)
        self.data = data
        self.bias = nm.random(0,1)

    def calculate_sigma():
        print("calculate (sigma of (input*weight)) + bias")
    
    def predict():
        print("apply the activation function")
    
    def get_prediction():
        print("get the prediction")
    
    def optimize():
        print("apply backpropagation")