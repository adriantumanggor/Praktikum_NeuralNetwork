### src/network/mlp.py
"""
Multi-Layer Perceptron implementation
"""
import random
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from ..network.activations import sigmoid, sigmoid_derivative

class MLP:
    """Multi-Layer Perceptron implementation from scratch"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.5, 
                 weight_init_range: Tuple[float, float] = (-1.0, 1.0),
                 bias_init_value: float = 0.0):
        """Initialize MLP with random weights and specified biases"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        min_w, max_w = weight_init_range
        self.weights_input_hidden = [[random.uniform(min_w, max_w) for _ in range(hidden_size)] 
                                   for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(min_w, max_w) for _ in range(output_size)] 
                                    for _ in range(hidden_size)]
        f
        # Initialize biases to the specified value
        self.bias_hidden = [bias_init_value] * hidden_size
        self.bias_output = [bias_init_value] * output_size
    
    def forward_pass(self, inputs: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Perform forward pass through the network
        Returns: (hidden_inputs, hidden_outputs, output_inputs, final_outputs)
        """
        # Calculate hidden layer
        hidden_inputs = []
        hidden_outputs = []
        
        for j in range(self.hidden_size):
            weighted_sum = sum(inputs[i] * self.weights_input_hidden[i][j] 
                             for i in range(self.input_size))
            weighted_sum += self.bias_hidden[j]
            hidden_inputs.append(weighted_sum)
            hidden_outputs.append(sigmoid(weighted_sum))
        
        # Calculate output layer
        output_inputs = []
        final_outputs = []
        
        for k in range(self.output_size):
            weighted_sum = sum(hidden_outputs[j] * self.weights_hidden_output[j][k] 
                             for j in range(self.hidden_size))
            weighted_sum += self.bias_output[k]
            output_inputs.append(weighted_sum)
            final_outputs.append(sigmoid(weighted_sum))
        
        return hidden_inputs, hidden_outputs, output_inputs, final_outputs
    
    def backward_pass(self, inputs: List[float], hidden_outputs: List[float], 
                     final_outputs: List[float], targets: List[float]) -> Dict[str, Any]:
        """
        Perform backpropagation and return detailed calculations
        """
        calculations = {
            'output_errors': [],
            'hidden_errors': [],
            'weight_updates': {
                'hidden_to_output': [],
                'input_to_hidden': []
            },
            'bias_updates': {
                'output': [],
                'hidden': []
            }
        }
        
        # Calculate output layer errors
        output_errors = []
        for k in range(self.output_size):
            error = (targets[k] - final_outputs[k]) * sigmoid_derivative(final_outputs[k])
            output_errors.append(error)
            calculations['output_errors'].append({
                'neuron': k,
                'target': targets[k],
                'prediction': final_outputs[k],
                'raw_error': targets[k] - final_outputs[k],
                'sigmoid_derivative': sigmoid_derivative(final_outputs[k]),
                'final_error': error
            })
        
        # Calculate hidden layer errors
        hidden_errors = []
        for j in range(self.hidden_size):
            error = sum(output_errors[k] * self.weights_hidden_output[j][k] 
                       for k in range(self.output_size))
            error *= sigmoid_derivative(hidden_outputs[j])
            hidden_errors.append(error)
            calculations['hidden_errors'].append({
                'neuron': j,
                'error_sum': sum(output_errors[k] * self.weights_hidden_output[j][k] 
                               for k in range(self.output_size)),
                'sigmoid_derivative': sigmoid_derivative(hidden_outputs[j]),
                'final_error': error
            })
        
        # Update weights and biases
        self._update_weights_and_biases(inputs, hidden_outputs, output_errors, 
                                      hidden_errors, calculations)
        
        return calculations
    
    def _update_weights_and_biases(self, inputs: List[float], hidden_outputs: List[float],
                                 output_errors: List[float], hidden_errors: List[float],
                                 calculations: Dict[str, Any]):
        """Update all weights and biases"""
        
        # Update hidden to output weights
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                old_weight = self.weights_hidden_output[j][k]
                gradient = self.learning_rate * output_errors[k] * hidden_outputs[j]
                new_weight = old_weight + gradient
                self.weights_hidden_output[j][k] = new_weight
                
                calculations['weight_updates']['hidden_to_output'].append({
                    'from_neuron': j,
                    'to_neuron': k,
                    'old_weight': old_weight,
                    'gradient': gradient,
                    'new_weight': new_weight
                })
        
        # Update input to hidden weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                old_weight = self.weights_input_hidden[i][j]
                gradient = self.learning_rate * hidden_errors[j] * inputs[i]
                new_weight = old_weight + gradient
                self.weights_input_hidden[i][j] = new_weight
                
                calculations['weight_updates']['input_to_hidden'].append({
                    'from_neuron': i,
                    'to_neuron': j,
                    'old_weight': old_weight,
                    'gradient': gradient,
                    'new_weight': new_weight
                })
        
        # Update output biases
        for k in range(self.output_size):
            old_bias = self.bias_output[k]
            gradient = self.learning_rate * output_errors[k]
            new_bias = old_bias + gradient
            self.bias_output[k] = new_bias
            
            calculations['bias_updates']['output'].append({
                'neuron': k,
                'old_bias': old_bias,
                'gradient': gradient,
                'new_bias': new_bias
            })
        
        # Update hidden biases
        for j in range(self.hidden_size):
            old_bias = self.bias_hidden[j]
            gradient = self.learning_rate * hidden_errors[j]
            new_bias = old_bias + gradient
            self.bias_hidden[j] = new_bias
            
            calculations['bias_updates']['hidden'].append({
                'neuron': j,
                'old_bias': old_bias,
                'gradient': gradient,
                'new_bias': new_bias
            })
    
    def calculate_loss(self, predictions: List[float], targets: List[float]) -> float:
        """
        Calculate Mean Squared Error loss.
        This version is robust against mixed types (list and numpy array).
        """
        # --- FIX: Konversi kedua input ke numpy array agar konsisten ---
        predictions_arr = np.array(predictions)
        targets_arr = np.array(targets)
        
        # Gunakan operasi numpy untuk menghitung error dan rata-ratanya
        squared_errors = np.square(predictions_arr - targets_arr.flatten())
        
        # np.mean akan menghitung rata-rata dan mengembalikan satu nilai float
        return np.mean(squared_errors)    
    
    def predict(self, inputs: List[float]) -> List[float]:
        """Make prediction for given inputs"""
        _, _, _, outputs = self.forward_pass(inputs)
        return outputs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for saving, ensuring JSON serializability."""
        # --- FIX: Konversi semua bobot dan bias ke list Python sebelum disimpan ---
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'weights_input_hidden': np.array(self.weights_input_hidden).tolist(),
            'weights_hidden_output': np.array(self.weights_hidden_output).tolist(),
            'bias_hidden': np.array(self.bias_hidden).tolist(),
            'bias_output': np.array(self.bias_output).tolist()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLP':
        """Create model from dictionary"""
        mlp = cls(
            input_size=data['input_size'],
            hidden_size=data['hidden_size'],
            output_size=data['output_size'],
            learning_rate=data['learning_rate']
        )
        mlp.weights_input_hidden = data['weights_input_hidden']
        mlp.weights_hidden_output = data['weights_hidden_output']
        mlp.bias_hidden = data['bias_hidden']
        mlp.bias_output = data['bias_output']
        return mlp