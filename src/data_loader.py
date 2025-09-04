# src/data_loader.py
"""
Data loader untuk training data gerbang AND dan OR
"""

import csv
import os

def get_and_gate_data():
    """Return training data untuk AND gate"""
    return [
        (0, 0, 0),  # x1, x2, expected_output
        (0, 1, 0),
        (1, 0, 0), 
        (1, 1, 1)
    ]

def get_or_gate_data():
    """Return training data untuk OR gate"""
    return [
        (0, 0, 0),  # x1, x2, expected_output
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1)
    ]

def create_training_data_files():
    """Buat file CSV untuk training data"""
    from config import DATA_DIR, AND_TRAINING_DATA, OR_TRAINING_DATA
    
    # Buat direktori jika belum ada
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # AND gate data
    and_data = get_and_gate_data()
    with open(AND_TRAINING_DATA, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'expected_output'])
        writer.writerows(and_data)
    
    # OR gate data  
    or_data = get_or_gate_data()
    with open(OR_TRAINING_DATA, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'expected_output'])
        writer.writerows(or_data)
    
    print(f"Training data files created:")
    print(f"- {AND_TRAINING_DATA}")
    print(f"- {OR_TRAINING_DATA}")

def load_training_data(filename):
    """Load training data dari CSV file"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x1 = int(row['x1'])
                x2 = int(row['x2'])
                expected = int(row['expected_output'])
                data.append((x1, x2, expected))
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan. Membuat file baru...")
        create_training_data_files()
        return load_training_data(filename)
    
    return data