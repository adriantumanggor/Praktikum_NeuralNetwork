### config.py
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Network configuration
NETWORK_CONFIG = {
    'input_size': 2,
    'hidden_size': 2,
    'output_size': 1,
    'learning_rate': 0.5,
    'weight_init_range': (-1.0, 1.0),
    'bias_init_value': 0.0
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 10000,
    'log_detailed_every': 1000,      # Log detailed setiap N epochs
    'log_first_epochs': 5,          # Log detailed untuk N epochs pertama
    'print_progress_every': 50,     # Print progress setiap N epochs
    'early_stopping_patience': 100, # Stop if no improvement for N epochs
    'target_loss': 0.01             # Target loss untuk early stopping
}

# Logging configuration
LOGGING_CONFIG = {
    'epoch_summary_file': 'epoch_summary.csv',
    'detailed_log_pattern': 'detailed_logs_epoch_{epoch}_sample_{sample}.csv',
    'model_save_pattern': 'model_epoch_{epoch}.json',
    'final_model_file': 'trained_model.json'
}

# Dataset configuration
DATASET_CONFIG = {
    'xor_dataset_file': 'xor_dataset.json',
}


