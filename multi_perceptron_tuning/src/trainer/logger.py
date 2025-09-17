# src/trainer/logger.py
"""
Logging functionality for training process
"""
import csv
import os
from typing import List, Dict, Any
import config

class TrainingLogger:
    """Handles all logging operations during training"""
    
    def __init__(self, logging_config: Dict[str, Any]):
        self.logging_config = logging_config
        self.epoch_logs = []
        self._setup_log_files()
    
    def _setup_log_files(self):
        """Setup log files and directories"""
        # Create epoch summary file
        self.epoch_summary_file = os.path.join(
            config.LOGS_DIR, 
            self.logging_config['epoch_summary_file']
        )
        
        # Write header for epoch summary
        with open(self.epoch_summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'average_loss', 'total_samples', 'best_loss_so_far'])
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, total_samples: int):
        """Log epoch summary information"""
        # Determine if this is the best loss so far
        best_loss = min([log.get('average_loss', float('inf')) for log in self.epoch_logs] + [avg_loss])
        
        # Store in memory
        epoch_data = {
            'epoch': epoch,
            'average_loss': avg_loss,
            'total_samples': total_samples,
            'best_loss_so_far': best_loss
        }
        self.epoch_logs.append(epoch_data)
        
        # Write to CSV
        with open(self.epoch_summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, total_samples, best_loss])
    
    def log_detailed_calculation(self, epoch: int, sample_idx: int, 
                               inputs: List[float], targets: List[float],
                               hidden_inputs: List[float], hidden_outputs: List[float],
                               output_inputs: List[float], final_outputs: List[float],
                               loss: float, calculations: Dict[str, Any]):
        """Log detailed calculations for a specific sample"""
        
        detailed_log_file = os.path.join(
            config.LOGS_DIR,
            self.logging_config['detailed_log_pattern'].format(
                epoch=epoch, sample=sample_idx
            )
        )
        
        # Prepare detailed log entries
        log_entries = []
        
        # Sample info
        log_entries.append({
            'step_type': 'sample_info',
            'epoch': epoch,
            'sample_index': sample_idx,
            'input_0': inputs[0] if len(inputs) > 0 else None,
            'input_1': inputs[1] if len(inputs) > 1 else None,
            'target_0': targets[0] if len(targets) > 0 else None,
            'description': f'Processing sample {sample_idx} in epoch {epoch}'
        })
        
        # Forward pass - Hidden layer
        for i, (h_input, h_output) in enumerate(zip(hidden_inputs, hidden_outputs)):
            log_entries.append({
                'step_type': 'forward_hidden',
                'neuron_index': i,
                'weighted_sum': h_input,
                'activation_output': h_output,
                'description': f'Hidden neuron {i} forward pass'
            })
        
        # Forward pass - Output layer
        for i, (o_input, o_output) in enumerate(zip(output_inputs, final_outputs)):
            log_entries.append({
                'step_type': 'forward_output',
                'neuron_index': i,
                'weighted_sum': o_input,
                'activation_output': o_output,
                'target': targets[i],
                'error': targets[i] - o_output,
                'description': f'Output neuron {i} forward pass'
            })
        
        # Loss calculation
        log_entries.append({
            'step_type': 'loss_calculation',
            'loss_value': loss,
            'description': f'MSE loss calculation'
        })
        
        # Backpropagation - Output errors
        for error_info in calculations['output_errors']:
            log_entries.append({
                'step_type': 'backprop_output_error',
                'neuron_index': error_info['neuron'],
                'target': error_info['target'],
                'prediction': error_info['prediction'],
                'raw_error': error_info['raw_error'],
                'sigmoid_derivative': error_info['sigmoid_derivative'],
                'final_error': error_info['final_error'],
                'description': f'Output neuron {error_info["neuron"]} error calculation'
            })
        
        # Backpropagation - Hidden errors
        for error_info in calculations['hidden_errors']:
            log_entries.append({
                'step_type': 'backprop_hidden_error',
                'neuron_index': error_info['neuron'],
                'error_sum': error_info['error_sum'],
                'sigmoid_derivative': error_info['sigmoid_derivative'],
                'final_error': error_info['final_error'],
                'description': f'Hidden neuron {error_info["neuron"]} error calculation'
            })
        
        # Weight updates
        for update_info in calculations['weight_updates']['input_to_hidden']:
            log_entries.append({
                'step_type': 'weight_update_input_hidden',
                'from_neuron': update_info['from_neuron'],
                'to_neuron': update_info['to_neuron'],
                'old_weight': update_info['old_weight'],
                'gradient': update_info['gradient'],
                'new_weight': update_info['new_weight'],
                'weight_change': update_info['new_weight'] - update_info['old_weight'],
                'description': f'Weight update: input {update_info["from_neuron"]} -> hidden {update_info["to_neuron"]}'
            })
        
        for update_info in calculations['weight_updates']['hidden_to_output']:
            log_entries.append({
                'step_type': 'weight_update_hidden_output',
                'from_neuron': update_info['from_neuron'],
                'to_neuron': update_info['to_neuron'],
                'old_weight': update_info['old_weight'],
                'gradient': update_info['gradient'],
                'new_weight': update_info['new_weight'],
                'weight_change': update_info['new_weight'] - update_info['old_weight'],
                'description': f'Weight update: hidden {update_info["from_neuron"]} -> output {update_info["to_neuron"]}'
            })
        
        # Bias updates
        for update_info in calculations['bias_updates']['hidden']:
            log_entries.append({
                'step_type': 'bias_update_hidden',
                'neuron_index': update_info['neuron'],
                'old_bias': update_info['old_bias'],
                'gradient': update_info['gradient'],
                'new_bias': update_info['new_bias'],
                'bias_change': update_info['new_bias'] - update_info['old_bias'],
                'description': f'Bias update: hidden neuron {update_info["neuron"]}'
            })
        
        for update_info in calculations['bias_updates']['output']:
            log_entries.append({
                'step_type': 'bias_update_output',
                'neuron_index': update_info['neuron'],
                'old_bias': update_info['old_bias'],
                'gradient': update_info['gradient'],
                'new_bias': update_info['new_bias'],
                'bias_change': update_info['new_bias'] - update_info['old_bias'],
                'description': f'Bias update: output neuron {update_info["neuron"]}'
            })
        
        # Write to CSV
        self._write_detailed_log_to_csv(detailed_log_file, log_entries)
    
    def _write_detailed_log_to_csv(self, filename: str, log_entries: List[Dict[str, Any]]):
        """Write detailed log entries to CSV file"""
        if not log_entries:
            return
        
        # Get all possible fieldnames from all entries
        fieldnames = set()
        for entry in log_entries:
            fieldnames.update(entry.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_entries)