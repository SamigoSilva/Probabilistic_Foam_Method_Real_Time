import numpy as np
from numba import njit
from typing import Tuple

@njit(fastmath=True, cache=True)
def update_potential_field(grid: np.ndarray, 
                         obstacles: list, 
                         robot_x: int, 
                         robot_y: int) -> np.ndarray:
    """
    Calcula o campo potencial usando método de gradiente descendente.
    Versão otimizada para Numba (sem closures ou yields).
    
    Args:
        grid: Matriz numpy do grid (será modificada)
        obstacles: Lista de tuplas (x,y) com posições dos obstáculos
        robot_x, robot_y: Posição atual do robô
        
    Returns:
        Grid atualizado
    """
    height, width = grid.shape
    new_grid = np.zeros_like(grid)
    
    # Constantes do campo potencial
    K_OBSTACLE = 0.8
    K_ROBOT = -0.3
    OBSTACLE_RADIUS = 3
    
    for y in range(height):
        for x in range(width):
            # Potencial base
            potential = 0.0
            
            # Influência dos obstáculos
            for ox, oy in obstacles:
                dist_sq = (x - ox)**2 + (y - oy)**2
                if dist_sq <= OBSTACLE_RADIUS**2:
                    potential += K_OBSTACLE * (1 - dist_sq / (OBSTACLE_RADIUS**2))
            
            # Influência do robô (atrativo)
            dist_to_robot = np.sqrt((x - robot_x)**2 + (y - robot_y)**2)
            if dist_to_robot > 0:
                potential += K_ROBOT / dist_to_robot
                
            new_grid[y, x] = np.clip(potential, -1, 1)
    
    return new_grid

# Função de teste compatível com Numba
@njit
def _test_potential():
    grid = np.zeros((50, 50), dtype=np.float32)
    obstacles = [(10, 10), (20, 20), (30, 30)]
    return update_potential_field(grid, obstacles, 25, 25)

if __name__ == "__main__":
    # Teste básico
    result = _test_potential()
    print("Campo potencial calculado com sucesso!")
    print("Valores mín/médio/máx:", result.min(), np.mean(result), result.max())
