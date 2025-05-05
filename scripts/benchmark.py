import time
from tqdm import tqdm
from core.foam import Foam

METHODS = ["potential", "occupancy", "neural"]

def run_benchmark(n_runs: int = 30, steps: int = 100):
    results = {method: [] for method in METHODS}
    
    for method in METHODS:
        for _ in tqdm(range(n_runs), desc=f"Testing {method}"):
            foam = Foam(width=64, height=64)
            start = time.time()
            
            for _ in range(steps):
                foam.update(method)
                
            results[method].append({
                'time': time.time() - start,
                'metrics': foam.get_metrics()
            })
    
    return results