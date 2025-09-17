### src/network/activations.py
"""
Activation functions and their derivatives
"""
import math

def sigmoid(x: float) -> float:
    """Sigmoid activation function"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid function (assumes x is already sigmoid output)"""
    return x * (1.0 - x)

def relu(x: float) -> float:
    """ReLU activation function"""
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    """Derivative of ReLU function"""
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    """Tanh activation function"""
    return math.tanh(x)

def tanh_derivative(x: float) -> float:
    """Derivative of tanh function (assumes x is already tanh output)"""
    return 1.0 - x * x
