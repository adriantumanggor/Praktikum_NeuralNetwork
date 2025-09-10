### src/data/dataset.py
"""
Dataset creation and handling
"""
import json
import os
from typing import List, Tuple
import config

class XORDataset:
    """XOR dataset creation and management"""
    
    def __init__(self):
        self.data = self._create_xor_data()
        self._save_dataset()
    
    def _create_xor_data(self) -> List[Tuple[List[float], List[float]]]:
        """Create XOR training dataset"""
        return [
            ([0.0, 0.0], [0.0]),  # 0 XOR 0 = 0
            ([0.0, 1.0], [1.0]),  # 0 XOR 1 = 1
            ([1.0, 0.0], [1.0]),  # 1 XOR 0 = 1
            ([1.0, 1.0], [0.0])   # 1 XOR 1 = 0
        ]
    
    def _save_dataset(self):
        """Save dataset to JSON file"""
        dataset_file = os.path.join(config.INPUT_DIR, config.DATASET_CONFIG['xor_dataset_file'])
        
        # Convert to JSON serializable format
        json_data = {
            'description': 'XOR Logic Gate Dataset',
            'input_size': 2,
            'output_size': 1,
            'samples': [
                {'input': inputs, 'target': targets} 
                for inputs, targets in self.data
            ]
        }
        
        with open(dataset_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def get_data(self) -> List[Tuple[List[float], List[float]]]:
        """Get training data"""
        return self.data.copy()
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'XORDataset':
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            json_data = json.load(f)
        
        dataset = cls.__new__(cls)
        dataset.data = [(sample['input'], sample['target']) 
                       for sample in json_data['samples']]
        return dataset
