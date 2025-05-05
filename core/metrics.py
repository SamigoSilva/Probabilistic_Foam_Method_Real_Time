import numpy as np

def calculate_metrics(grid: np.ndarray) -> dict:
    """Calcula métricas científicas consolidadas"""
    coverage = np.mean(grid > 0.5)
    entropy = -np.sum(grid * np.log2(grid + 1e-9))
    
    return {
        'coverage': float(coverage),
        'entropy': float(entropy),
        'uniformity': float(1 - np.std(grid)/0.5)
    }