
# analysis/analyzer.py
"""
Training log analysis tools
"""
import csv
import glob
import os
from typing import List, Dict, Any, Tuple
import config
from analysis.visualizer import TrainingVisualizer

class TrainingAnalyzer:
    """Analyze training logs and provide insights"""
    
    def __init__(self):
        self.visualizer = TrainingVisualizer()
    
    def analyze_training_session(self) -> Dict[str, Any]:
        """Comprehensive analysis of the training session"""
        print("=== Training Session Analysis ===")
        
        # Load and analyze epoch logs
        summary = self.visualizer.create_training_summary()
        
        if not summary:
            print("No training data found for analysis")
            return {}
        
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Initial Loss: {summary['initial_loss']:.6f}")
        print(f"Final Loss: {summary['final_loss']:.6f}")
        print(f"Best Loss: {summary['best_loss']:.6f} (reached at epoch {summary['convergence_epoch']})")
        print(f"Loss Reduction: {summary['loss_reduction']:.6f} ({summary['loss_reduction_percentage']:.2f}%)")
        print(f"Average Loss: {summary['average_loss']:.6f}")
        
        # Analyze detailed logs
        detailed_analysis = self._analyze_detailed_logs()
        summary.update(detailed_analysis)
        
        return summary
    
    def _analyze_detailed_logs(self) -> Dict[str, Any]:
        """Analyze detailed calculation logs"""
        print("\n=== Detailed Logs Analysis ===")
        
        log_files = glob.glob(os.path.join(config.LOGS_DIR, 'detailed_logs_*.csv'))
        
        if not log_files:
            print("No detailed logs found")
            return {}
        
        print(f"Found {len(log_files)} detailed log files")
        
        # Analyze a sample of detailed logs
        sample_file = log_files[0]  # Analyze first file as example
        
        analysis = self._analyze_single_detailed_log(sample_file)
        
        return {
            'detailed_logs_count': len(log_files),
            'sample_analysis': analysis
        }
    
    def _analyze_single_detailed_log(self, log_file: str) -> Dict[str, Any]:
        """Analyze a single detailed log file"""
        print(f"\nAnalyzing: {os.path.basename(log_file)}")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            logs = list(reader)
        
        if not logs:
            return {}
        
        # Count operation types
        operation_counts = {}
        weight_changes = []
        bias_changes = []
        
        for log in logs:
            step_type = log.get('step_type', 'unknown')
            operation_counts[step_type] = operation_counts.get(step_type, 0) + 1
            
            # Collect weight changes
            if 'weight_change' in log and log['weight_change']:
                try:
                    weight_changes.append(float(log['weight_change']))
                except ValueError:
                    pass
            
            # Collect bias changes
            if 'bias_change' in log and log['bias_change']:
                try:
                    bias_changes.append(float(log['bias_change']))
                except ValueError:
                    pass
        
        analysis = {
            'total_operations': len(logs),
            'operation_breakdown': operation_counts,
            'weight_updates': len(weight_changes),
            'bias_updates': len(bias_changes)
        }
        
        if weight_changes:
            analysis['weight_change_stats'] = {
                'mean': sum(weight_changes) / len(weight_changes),
                'max': max(weight_changes),
                'min': min(weight_changes),
                'abs_mean': sum(abs(w) for w in weight_changes) / len(weight_changes)
            }
        
        if bias_changes:
            analysis['bias_change_stats'] = {
                'mean': sum(bias_changes) / len(bias_changes),
                'max': max(bias_changes),
                'min': min(bias_changes),
                'abs_mean': sum(abs(b) for b in bias_changes) / len(bias_changes)
            }
        
        print(f"  Total operations: {analysis['total_operations']}")
        print("  Operation breakdown:")
        for op_type, count in operation_counts.items():
            print(f"    {op_type}: {count}")
        
        return analysis
    
    def generate_full_report(self, output_file: str = None):
        """Generate comprehensive training report"""
        if output_file is None:
            output_file = os.path.join(config.RESULTS_DIR, 'training_report.txt')
        
        # Perform analysis
        analysis = self.analyze_training_session()
        
        # Generate visualizations
        plot_file = os.path.join(config.PLOTS_DIR, 'training_analysis.png')
        self.visualizer.plot_learning_progress(plot_file, show_plot=False)
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("MLP Training Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            if analysis:
                f.write(f"Training completed in {analysis.get('total_epochs', 'N/A')} epochs\n")
                f.write(f"Initial loss: {analysis.get('initial_loss', 'N/A'):.6f}\n")
                f.write(f"Final loss: {analysis.get('final_loss', 'N/A'):.6f}\n")
                f.write(f"Best loss: {analysis.get('best_loss', 'N/A'):.6f}\n")
                f.write(f"Loss reduction: {analysis.get('loss_reduction_percentage', 'N/A'):.2f}%\n\n")
                
                if 'detailed_logs_count' in analysis:
                    f.write(f"Detailed logs generated: {analysis['detailed_logs_count']} files\n")
                
                f.write(f"\nVisualization saved to: {plot_file}\n")
            
            else:
                f.write("No training data available for analysis\n")
        
        print(f"\nFull report saved to: {output_file}")
        return output_file


def main():
    """Main analysis function"""
    analyzer = TrainingAnalyzer()
    analyzer.generate_full_report()

if __name__ == "__main__":
    main()