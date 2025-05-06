import numpy as np
from numba import njit
from typing import List, Tuple

@njit(fastmath=True, cache=True)
def _clip_scalar(value: float, min_val: float, max_val: float) -> float:
    """Versão do clip compatível com Numba para valores escalares"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

@njit(fastmath=True, cache=True)
def update_potential_field(grid: np.ndarray, 
                         obstacles: List[Tuple[int, int]], 
                         robot_x: int, 
                         robot_y: int) -> np.ndarray:
    """
    Calcula o campo potencial usando método de gradiente descendente.
    Versão 100% compatível com Numba.
    """
    height, width = grid.shape
    new_grid = np.zeros_like(grid)
    
    # Constantes do campo potencial (agora como floats explícitos)
    K_OBSTACLE = 0.8
    K_ROBOT = -0.3
    OBSTACLE_RADIUS = 3.0
    MIN_VAL = -1.0
    MAX_VAL = 1.0
    
    for y in range(height):
        for x in range(width):
            potential = 0.0
            
            # Influência dos obstáculos
            for ox, oy in obstacles:
                dist_sq = (x - ox)**2 + (y - oy)**2
                if dist_sq <= OBSTACLE_RADIUS**2:
                    potential += K_OBSTACLE * (1.0 - dist_sq / (OBSTACLE_RADIUS**2))
            
            # Influência do robô
            dist_to_robot = np.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            if dist_to_robot > 0:
                potential += K_ROBOT / dist_to_robot
                
            new_grid[y, x] = _clip_scalar(potential, MIN_VAL, MAX_VAL)
    
    return new_grid

@njit
def _test_potential():
    """Função de teste totalmente compatível com Numba"""
    grid = np.zeros((50, 50), dtype=np.float32)
    obstacles = [(10, 10), (20, 20), (30, 30)]
    return update_potential_field(grid, obstacles, 25, 25)

if __name__ == "__main__":
    # Teste que agora funciona com Numba
    result = _test_potential()
    print("Teste concluído com sucesso!")
    print("Valor mínimo:", np.min(result))
    print("Valor médio:", np.mean(result))
    print("Valor máximo:", np.max(result))
