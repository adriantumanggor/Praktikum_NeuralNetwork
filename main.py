# main.py
"""
Entry point untuk training single perceptron pada gerbang AND dan OR
"""

import csv
import os
from src.data_loader import get_and_gate_data, get_or_gate_data, create_training_data_files
from src.trainer import PerceptronTrainer
from config import AND_LOG_FILE, OR_LOG_FILE, SUMMARY_FILE, SUMMARY_HEADERS, RESULTS_DIR

def save_summary_report(and_summary, or_summary):
    """Save summary report ke CSV"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADERS)
        writer.writeheader()
        writer.writerow(and_summary)
        writer.writerow(or_summary)
    
    print(f"\nSummary report saved to: {SUMMARY_FILE}")

def print_comparison_report(and_summary, or_summary):
    """Print perbandingan hasil training AND vs OR gate"""
    print("\n" + "="*60)
    print("COMPARISON REPORT: AND vs OR Gate Training")
    print("="*60)
    
    print(f"{'Metric':<25} {'AND Gate':<15} {'OR Gate':<15}")
    print("-" * 55)
    print(f"{'Epochs to Converge':<25} {and_summary['epochs_to_converge']:<15} {or_summary['epochs_to_converge']:<15}")
    print(f"{'Final Accuracy':<25} {and_summary['final_accuracy']:<15.2%} {or_summary['final_accuracy']:<15.2%}")
    print(f"{'Total Weight Updates':<25} {and_summary['total_weight_updates']:<15} {or_summary['total_weight_updates']:<15}")
    print(f"{'Converged':<25} {'Yes' if and_summary['converged'] else 'No':<15} {'Yes' if or_summary['converged'] else 'No':<15}")
    
    print("\nFinal Weights:")
    print(f"{'Gate':<10} {'w1':<10} {'w2':<10} {'w_bias':<10}")
    print("-" * 40)
    print(f"{'AND':<10} {and_summary['final_w1']:<10} {and_summary['final_w2']:<10} {and_summary['final_w_bias']:<10}")
    print(f"{'OR':<10} {or_summary['final_w1']:<10} {or_summary['final_w2']:<10} {or_summary['final_w_bias']:<10}")

def main():
    """Main function"""
    print("Single Perceptron Neural Network - AND & OR Gates")
    print("=" * 55)
    
    # Create training data files
    create_training_data_files()
    
    # Get training data
    and_data = get_and_gate_data()
    or_data = get_or_gate_data()
    
    # Initialize trainer
    trainer = PerceptronTrainer()
    
    # Train AND Gate
    and_summary = trainer.train(and_data, "AND", AND_LOG_FILE)
    trainer.test_final_model(and_data, "AND")
    
    # Train OR Gate
    or_summary = trainer.train(or_data, "OR", OR_LOG_FILE)  
    trainer.test_final_model(or_data, "OR")
    
    # Save summary report
    save_summary_report(and_summary, or_summary)
    
    # Print comparison
    print_comparison_report(and_summary, or_summary)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("Generated Files:")
    print(f"- {AND_LOG_FILE} (detailed AND gate training log)")
    print(f"- {OR_LOG_FILE} (detailed OR gate training log)")  
    print(f"- {SUMMARY_FILE} (training summary)")
    print("\nCheck the CSV files for detailed epoch-by-epoch analysis!")

if __name__ == "__main__":
    main()