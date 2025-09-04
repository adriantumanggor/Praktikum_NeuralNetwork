# src/perceptron.py
import random
from config import LEARNING_RATE, WEIGHT_INIT_RANGE

class Perceptron:
    def __init__(self):
        """Initialize perceptron dengan random weights"""
        self.learning_rate = LEARNING_RATE
        self.w1 = random.uniform(*WEIGHT_INIT_RANGE)  # Weight untuk input x1
        self.w2 = random.uniform(*WEIGHT_INIT_RANGE)  # Weight untuk input x2  
        self.w_bias = random.uniform(*WEIGHT_INIT_RANGE)  # Weight untuk bias
        self.bias = 1  # Bias input selalu 1
        
    def step_function(self, x):
        """Step activation function dengan threshold = 0"""
        return 1 if x >= 0 else 0
    
    def forward(self, x1, x2):
        """Forward pass - hitung output perceptron"""
        weighted_sum = (self.w1 * x1) + (self.w2 * x2) + (self.w_bias * self.bias)
        predicted_output = self.step_function(weighted_sum)
        return weighted_sum, predicted_output
    
    def train_sample(self, x1, x2, expected_output):
        """
        Train dengan satu sample menggunakan perceptron learning rule
        Return: (predicted_output, error, weight_updated, weighted_sum)
        """
        # Forward pass
        weighted_sum, predicted_output = self.forward(x1, x2)
        
        # Hitung error
        error = expected_output - predicted_output
        
        # Update weights jika ada error
        weight_updated = False
        if error != 0:
            self.w1 += self.learning_rate * error * x1
            self.w2 += self.learning_rate * error * x2
            self.w_bias += self.learning_rate * error * self.bias
            weight_updated = True
            
        return predicted_output, error, weight_updated, weighted_sum
    
    def predict(self, x1, x2):
        """Prediksi output tanpa training"""
        _, predicted_output = self.forward(x1, x2)
        return predicted_output
    
    def get_weights(self):
        """Return current weights"""
        return self.w1, self.w2, self.w_bias
    
    def evaluate_accuracy(self, training_data):
        """Evaluasi accuracy pada training data"""
        correct = 0
        total = len(training_data)
        
        for x1, x2, expected in training_data:
            predicted = self.predict(x1, x2)
            if predicted == expected:
                correct += 1
                
        return correct / total if total > 0 else 0.0