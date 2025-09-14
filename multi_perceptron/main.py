### main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer.trainer import MLPTrainer
from src.data.dataset import XORDataset
import config

def main():
    """Main training function"""
    print("=== MLP Manual Training untuk XOR Problem ===")
    print("Menggunakan struktur project yang terorganisir")
    print()
    
    dataset = XORDataset()
    training_data = dataset.get_data()
    
    # Initialize trainer
    trainer = MLPTrainer(
        network_config=config.NETWORK_CONFIG,
        training_config=config.TRAINING_CONFIG,
        logging_config=config.LOGGING_CONFIG
    )
    
    # Start training
    print("Memulai training...")
    trainer.train(training_data)
    
    # Test trained model
    print("\n=== Testing Trained Network ===")
    trainer.test(training_data)
    
    # Save final model
    trainer.save_model()
    
    print(f"\nTraining selesai! Check results di: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
