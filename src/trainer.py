# src/trainer.py
"""
Training logic dengan detailed CSV logging
"""

import csv
import os
from src.perceptron import Perceptron
from config import MAX_EPOCHS, LOG_HEADERS, RESULTS_DIR

class PerceptronTrainer:
    def __init__(self):
        self.perceptron = None
        self.training_log = []
        
    def train(self, training_data, gate_type, log_file):
        """
        Train perceptron dengan detailed logging
        """
        # Initialize perceptron
        self.perceptron = Perceptron()
        self.training_log = []
        
        # Buat direktori results jika belum ada
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print(f"\n=== Training {gate_type} Gate ===")
        print(f"Initial weights: w1={self.perceptron.w1:.3f}, w2={self.perceptron.w2:.3f}, w_bias={self.perceptron.w_bias:.3f}")
        
        converged = False
        total_weight_updates = 0
        
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_errors = 0
            epoch_weight_updates = 0
            
            # Train dengan semua samples dalam epoch ini
            for sample_idx, (x1, x2, expected_output) in enumerate(training_data, 1):
                # Get current weights sebelum training
                w1, w2, w_bias = self.perceptron.get_weights()
                
                # Train sample
                predicted_output, error, weight_updated, weighted_sum = self.perceptron.train_sample(x1, x2, expected_output)
                
                if weight_updated:
                    epoch_weight_updates += 1
                    total_weight_updates += 1
                
                if error != 0:
                    epoch_errors += 1
                
                # Log detail training step
                log_entry = {
                    'epoch': epoch,
                    'sample_idx': sample_idx,
                    'x1': x1,
                    'x2': x2,
                    'bias': self.perceptron.bias,
                    'w1': round(w1, 4),
                    'w2': round(w2, 4),
                    'w_bias': round(w_bias, 4),
                    'weighted_sum': round(weighted_sum, 4),
                    'predicted_output': predicted_output,
                    'expected_output': expected_output,
                    'error': error,
                    'weight_updated': weight_updated,
                    'converged': False  # Will be updated after epoch check
                }
                
                self.training_log.append(log_entry)
            
            # Check convergence (semua samples benar dalam epoch ini)
            if epoch_errors == 0:
                converged = True
                # Update converged status untuk semua entries di epoch ini
                for i in range(-len(training_data), 0):
                    self.training_log[i]['converged'] = True
                
                print(f"Converged at epoch {epoch}!")
                break
            
            # Progress report setiap 50 epoch
            if epoch % 50 == 0:
                accuracy = self.perceptron.evaluate_accuracy(training_data)
                print(f"Epoch {epoch}: {epoch_errors} errors, accuracy={accuracy:.2%}")
        
        # Final results
        final_w1, final_w2, final_w_bias = self.perceptron.get_weights()
        final_accuracy = self.perceptron.evaluate_accuracy(training_data)
        epochs_to_converge = epoch if converged else MAX_EPOCHS
        
        print(f"Training completed!")
        print(f"Epochs to converge: {epochs_to_converge}")
        print(f"Final weights: w1={final_w1:.4f}, w2={final_w2:.4f}, w_bias={final_w_bias:.4f}")
        print(f"Final accuracy: {final_accuracy:.2%}")
        print(f"Total weight updates: {total_weight_updates}")
        
        # Save training log to CSV
        self.save_training_log(log_file)
        
        # Return summary info
        return {
            'gate_type': gate_type,
            'epochs_to_converge': epochs_to_converge,
            'final_w1': round(final_w1, 4),
            'final_w2': round(final_w2, 4),
            'final_w_bias': round(final_w_bias, 4),
            'final_accuracy': final_accuracy,
            'total_weight_updates': total_weight_updates,
            'converged': converged
        }
    
    def save_training_log(self, log_file):
        """Save detailed training log ke CSV"""
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(self.training_log)
        
        print(f"Training log saved to: {log_file}")
        print(f"Total logged entries: {len(self.training_log)}")
    
    def test_final_model(self, training_data, gate_type):
        """Test final model dan tampilkan hasil"""
        print(f"\n=== Testing Final {gate_type} Model ===")
        print("Input | Expected | Predicted | Correct")
        print("------|----------|-----------|--------")
        
        for x1, x2, expected in training_data:
            predicted = self.perceptron.predict(x1, x2)
            correct = "✓" if predicted == expected else "✗"
            print(f"  {x1},{x2} |    {expected}     |     {predicted}     |   {correct}")