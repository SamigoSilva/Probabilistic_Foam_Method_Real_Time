import time
import numpy as np
from core.foam import Foam
from typing import Dict, List

def benchmark_simulation(methods: List[str], grid_sizes: List[int]) -> Dict[str, List[float]]:
    """Executa benchmark para diferentes métodos e tamanhos de grid"""
    results = {'method': [], 'grid_size': [], 'fps': [], 'ram_mb': []}
    
    for method in methods:
        for size in grid_sizes:
            # Configuração inicial
            foam = Foam()
            foam.config.update_config("grid.width", size)
            foam.config.update_config("grid.height", size)
            foam.config.update_config("methods.default", method)
            
            # Warm-up
            for _ in range(10):
                foam.update()
            
            # Medição principal
            start_time = time.time()
            frames = 0
            for _ in range(100):  # 100 iterações para média estável
                foam.update()
                frames += 1
                
            elapsed = time.time() - start_time
            fps = frames / elapsed
            
            # Armazena resultados (NOTA: Aqui está o return corrigido)
            results['method'].append(method)
            results['grid_size'].append(f"{size}x{size}")
            results['fps'].append(fps)
            
    return results  # Este deve estar alinhado com o início da função

if __name__ == "__main__":
    # Exemplo de uso
    methods_to_test = ["potential", "neural"]
    grid_sizes_to_test = [60, 80]
    
    benchmark_results = benchmark_simulation(methods_to_test, grid_sizes_to_test)
    print("Resultados do Benchmark:")
    for i in range(len(benchmark_results['method'])):
        print(f"{benchmark_results['method'][i]} {benchmark_results['grid_size'][i]}: {benchmark_results['fps'][i]:.1f} FPS")
