
### src/trainer/trainer.py
"""
Training logic for MLP
"""
import os
import json
from typing import List, Tuple, Dict, Any
from ..network.mlp import MLP
from ..trainer.logger import TrainingLogger
import config

class MLPTrainer:
    """Handles MLP training process"""
    
    def __init__(self, network_config: Dict[str, Any], 
                 training_config: Dict[str, Any],
                 logging_config: Dict[str, Any]):
        
        self.network_config = network_config
        self.training_config = training_config
        self.logging_config = logging_config
        
        # Initialize network
        self.mlp = MLP(**network_config)
        
        # Initialize logger
        self.logger = TrainingLogger(logging_config)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train(self, training_data: List[Tuple[List[float], List[float]]]):
        """Main training loop"""
        print(f"Training dimulai dengan {len(training_data)} samples")
        print(f"Network: {self.network_config['input_size']} -> {self.network_config['hidden_size']} -> {self.network_config['output_size']}")
        print(f"Learning rate: {self.network_config['learning_rate']}")
        print()
        
        for epoch in range(self.training_config['epochs']):
            self.current_epoch = epoch
            
            # Determine if we should log detailed calculations
            should_log_detailed = self._should_log_detailed(epoch)
            
            # Train one epoch
            avg_loss = self._train_epoch(training_data, epoch, should_log_detailed)
            
            # Log epoch summary
            self.logger.log_epoch_summary(epoch, avg_loss, len(training_data))
            
            # Print progress
            if epoch % self.training_config['print_progress_every'] == 0:
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")
            
            # Check for improvement
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save model periodically
            if epoch % self.training_config['save_model_every'] == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self._should_stop_early(avg_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Target loss {self.training_config['target_loss']} reached!")
                break
            
            if self.epochs_without_improvement >= self.training_config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"No improvement for {self.training_config['early_stopping_patience']} epochs")
                break
        
        print(f"\nTraining completed! Best loss: {self.best_loss:.6f}")
    
    def _train_epoch(self, training_data: List[Tuple[List[float], List[float]]], 
                    epoch: int, log_detailed: bool) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        
        for sample_idx, (inputs, targets) in enumerate(training_data):
            # Forward pass
            hidden_inputs, hidden_outputs, output_inputs, final_outputs = self.mlp.forward_pass(inputs)
            
            # Calculate loss
            loss = self.mlp.calculate_loss(final_outputs, targets)
            total_loss += loss
            
            # Backward pass
            calculations = self.mlp.backward_pass(inputs, hidden_outputs, final_outputs, targets)
            
            # Log detailed calculations if needed
            if log_detailed:
                self.logger.log_detailed_calculation(
                    epoch, sample_idx, inputs, targets, 
                    hidden_inputs, hidden_outputs, output_inputs, final_outputs,
                    loss, calculations
                )
        
        return total_loss / len(training_data)
    
    def _should_log_detailed(self, epoch: int) -> bool:
        """Determine if we should log detailed calculations"""
        return (epoch < self.training_config['log_first_epochs'] or 
                epoch % self.training_config['log_detailed_every'] == 0)
    
    def _should_stop_early(self, current_loss: float) -> bool:
        """Check if we should stop training early"""
        return current_loss <= self.training_config['target_loss']
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_file = os.path.join(
            config.MODELS_DIR, 
            self.logging_config['model_save_pattern'].format(epoch=epoch)
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump(self.mlp.to_dict(), f, indent=2)
    
    def save_model(self):
        """Save final trained model"""
        model_file = os.path.join(config.MODELS_DIR, self.logging_config['final_model_file'])
        
        with open(model_file, 'w') as f:
            json.dump(self.mlp.to_dict(), f, indent=2)
        
        print(f"Model saved to: {model_file}")
    
    def test(self, test_data: List[Tuple[List[float], List[float]]]):
        """Test the trained network"""
        print("Input\t| Expected | Predicted | Error")
        print("-" * 40)
        
        for inputs, expected in test_data:
            prediction = self.mlp.predict(inputs)
            error = abs(expected[0] - prediction[0])
            
            print(f"{inputs}\t| {expected[0]:.4f}   | {prediction[0]:.4f}    | {error:.4f}")
