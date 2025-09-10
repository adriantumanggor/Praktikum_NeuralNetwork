# src/__init__.py
"""
Single Perceptron Neural Network Package
Implementasi perceptron untuk gerbang logika AND dan OR
"""

__version__ = "1.0.0"
__author__ = "Perceptron Project"

from .perceptron import Perceptron
from .trainer import PerceptronTrainer
from .data_loader import get_and_gate_data, get_or_gate_data, load_training_data

__all__ = [
    'Perceptron',
    'PerceptronTrainer', 
    'get_and_gate_data',
    'get_or_gate_data',
    'load_training_data'
]