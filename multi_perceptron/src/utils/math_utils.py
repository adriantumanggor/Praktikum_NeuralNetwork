
# ================================================================
# src/utils/math_utils.py
# ================================================================
"""
Mathematical utility functions
"""
import math
from typing import List, Tuple

def mean_squared_error(predictions: List[float], targets: List[float]) -> float:
    """Calculate Mean Squared Error"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

def root_mean_squared_error(predictions: List[float], targets: List[float]) -> float:
    """Calculate Root Mean Squared Error"""
    return math.sqrt(mean_squared_error(predictions, targets))

def mean_absolute_error(predictions: List[float], targets: List[float]) -> float:
    """Calculate Mean Absolute Error"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)

def accuracy_threshold(predictions: List[float], targets: List[float], threshold: float = 0.5) -> float:
    """Calculate accuracy using threshold (for binary classification)"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    correct = sum(1 for p, t in zip(predictions, targets) 
                  if (p >= threshold) == (t >= threshold))
    return correct / len(predictions)

def normalize_list(values: List[float]) -> List[float]:
    """Normalize list of values to [0, 1] range"""
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [0.0] * len(values)
    
    return [(v - min_val) / (max_val - min_val) for v in values]