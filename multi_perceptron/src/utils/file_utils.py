# src/utils/file_utils.py
"""
File and directory utilities
"""
import os
import json
from typing import List, Dict, Any
import config

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_directory_exists(filepath: str):
    """Ensure the directory for a file exists"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


