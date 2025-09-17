
# analysis/visualizer.py
"""
Training visualization tools
"""
import csv
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
import config

class TrainingVisualizer:
    """Create visualizations from training logs"""
    
    def __init__(self):
        self.epoch_logs = []
    
    def load_epoch_logs(self):
        """Load epoch logs from CSV file"""
        epoch_file = os.path.join(config.LOGS_DIR, 'epoch_summary.csv')
        
        if not os.path.exists(epoch_file):
            print(f"Epoch log file not found: {epoch_file}")
            return
        
        self.epoch_logs = []
        with open(epoch_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.epoch_logs.append({
                    'epoch': int(row['epoch']),
                    'average_loss': float(row['average_loss']),
                    'total_samples': int(row['total_samples']),
                    'best_loss_so_far': float(row['best_loss_so_far'])
                })
    
    def plot_training_curve(self, save_path: str = None, show_plot: bool = True):
        """Plot training loss curve"""
        if not self.epoch_logs:
            self.load_epoch_logs()
        
        if not self.epoch_logs:
            print("No epoch logs available for plotting")
            return
        
        epochs = [log['epoch'] for log in self.epoch_logs]
        losses = [log['average_loss'] for log in self.epoch_logs]
        best_losses = [log['best_loss_so_far'] for log in self.epoch_logs]
        
        plt.figure(figsize=(12, 8))
        
        # Main loss curve
        plt.subplot(2, 1, 1)
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        plt.plot(epochs, best_losses, 'r--', linewidth=1, label='Best Loss So Far')
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale for better visibility
        plt.subplot(2, 1, 2)
        plt.semilogy(epochs, losses, 'b-', linewidth=2, label='Training Loss (Log Scale)')
        plt.title('Training Loss Over Time (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curve saved to: {save_path}")
        
        if show_plot:
            plt.show()
    
    def plot_learning_progress(self, save_path: str = None, show_plot: bool = True):
        """Plot learning progress with additional metrics"""
        if not self.epoch_logs:
            self.load_epoch_logs()
        
        if not self.epoch_logs:
            print("No epoch logs available for plotting")
            return
        
        epochs = [log['epoch'] for log in self.epoch_logs]
        losses = [log['average_loss'] for log in self.epoch_logs]
        
        # Calculate improvement rate
        improvement_rates = []
        for i in range(1, len(losses)):
            rate = losses[i-1] - losses[i]
            improvement_rates.append(rate)
        
        plt.figure(figsize=(15, 10))
        
        # Loss curve
        plt.subplot(2, 2, 1)
        plt.plot(epochs, losses, 'b-', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Log scale loss
        plt.subplot(2, 2, 2)
        plt.semilogy(epochs, losses, 'g-', linewidth=2)
        plt.title('Training Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log)')
        plt.grid(True, alpha=0.3)
        
        # Improvement rate
        plt.subplot(2, 2, 3)
        if len(improvement_rates) > 0:
            plt.plot(epochs[1:], improvement_rates, 'r-', linewidth=1, alpha=0.7)
            plt.title('Loss Improvement Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Reduction')
            plt.grid(True, alpha=0.3)
        
        # Final convergence (last 20% of training)
        plt.subplot(2, 2, 4)
        convergence_start = max(1, int(len(epochs) * 0.8))
        conv_epochs = epochs[convergence_start:]
        conv_losses = losses[convergence_start:]
        
        if len(conv_epochs) > 0:
            plt.plot(conv_epochs, conv_losses, 'purple', linewidth=2)
            plt.title('Convergence (Last 20% of Training)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning progress plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
    
    def create_training_summary(self) -> Dict[str, Any]:
        """Create summary statistics from training"""
        if not self.epoch_logs:
            self.load_epoch_logs()
        
        if not self.epoch_logs:
            return {}
        
        losses = [log['average_loss'] for log in self.epoch_logs]
        
        summary = {
            'total_epochs': len(self.epoch_logs),
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'worst_loss': max(losses),
            'loss_reduction': losses[0] - losses[-1],
            'loss_reduction_percentage': ((losses[0] - losses[-1]) / losses[0]) * 100,
            'convergence_epoch': losses.index(min(losses)),
            'average_loss': sum(losses) / len(losses)
        }
        
        return summary
