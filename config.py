# config.py
"""
Konfigurasi parameter untuk training perceptron
"""

# Training parameters
LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
WEIGHT_INIT_RANGE = (-0.5, 0.5)

# File paths
DATA_DIR = "data"
RESULTS_DIR = "data/results"
AND_TRAINING_DATA = f"{DATA_DIR}/and_gate_training.csv"
OR_TRAINING_DATA = f"{DATA_DIR}/or_gate_training.csv"
AND_LOG_FILE = f"{RESULTS_DIR}/and_training_log.csv"
OR_LOG_FILE = f"{RESULTS_DIR}/or_training_log.csv"
SUMMARY_FILE = f"{RESULTS_DIR}/training_summary.csv"

# CSV Headers
LOG_HEADERS = [
    "epoch", "sample_idx", "x1", "x2", "bias", 
    "w1", "w2", "w_bias", "weighted_sum", 
    "predicted_output", "expected_output", "error", 
    "weight_updated", "converged"
]

SUMMARY_HEADERS = [
    "gate_type", "epochs_to_converge", "final_w1", 
    "final_w2", "final_w_bias", "final_accuracy", 
    "total_weight_updates", "converged"
]