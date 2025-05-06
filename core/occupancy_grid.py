# core/occupancy_grid.py
import numpy as np
from numba import njit
from typing import Tuple, Optional
import logging
import math

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit(fastmath=True, cache=True)
def update_occupancy(grid: np.ndarray, 
                    sensor_data: np.ndarray,
                    robot_pos: Tuple[int, int] = (0, 0),
                    sensor_range: float = 5.0) -> np.ndarray:
    """
    Atualiza a grade de ocupação usando o modelo de sensor inverso.
    
    Args:
        grid: Grade 2D de probabilidades (0 a 1)
        sensor_data: Leitura do sensor (array 1D de distâncias em metros)
        robot_pos: Posição atual do robô (x, y) em células
        sensor_range: Alcance máximo do sensor em metros
        
    Returns:
        Grade atualizada com as novas probabilidades
    """
    # Parâmetros do modelo
    P_OCCUPIED = 0.9  # Probabilidade se célula ocupada
    P_FREE = 0.3      # Probabilidade se célula livre
    P_PRIOR = 0.5     # Probabilidade inicial
    
    # Cria uma cópia para não modificar o original
    new_grid = grid.copy()
    height, width = grid.shape
    
    # Converte posição do robô para inteiros
    rx, ry = int(robot_pos[0]), int(robot_pos[1])
    
    # Processa cada medida do sensor
    for angle_idx in range(len(sensor_data)):
        distance = sensor_data[angle_idx]
        
        # Converte ângulo para radianos (assumindo leituras equidistantes)
        angle = 2 * math.pi * angle_idx / len(sensor_data)
        
        # Calcula posição final do raio
        end_x = rx + int(distance * math.cos(angle))
        end_y = ry + int(distance * math.sin(angle))
        
        # Traça uma linha do robô até o obstáculo (algoritmo de Bresenham)
        x, y = rx, ry
        dx = abs(end_x - x)
        dy = -abs(end_y - y)
        sx = 1 if x < end_x else -1
        sy = 1 if y < end_y else -1
        err = dx + dy
        
        while True:
            # Atualiza probabilidade
            if 0 <= x < width and 0 <= y < height:
                if x == end_x and y == end_y:  # Célula ocupada
                    new_grid[y, x] = (P_OCCUPIED * new_grid[y, x]) / (
                        P_OCCUPIED * new_grid[y, x] + (1 - P_OCCUPIED) * (1 - new_grid[y, x]))
                else:  # Célula livre
                    new_grid[y, x] = (P_FREE * new_grid[y, x]) / (
                        P_FREE * new_grid[y, x] + (1 - P_FREE) * (1 - new_grid[y, x]))
            
            if x == end_x and y == end_y:
                break
                
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
    
    return new_grid

def test_occupancy():
    """Função de teste para verificar a implementação"""
    grid = np.full((50, 50), 0.5)  # Grade 50x50 com probabilidade inicial 0.5
    
    # Simula leitura do sensor (4 medidas: frente, direita, trás, esquerda)
    sensor_data = np.array([3.0, 2.5, 1.0, 4.0])  # Distâncias em metros
    
    updated = update_occupancy(grid, sensor_data, robot_pos=(25, 25))
    
    print("Grade atualizada - Valor médio:", np.mean(updated))
    return updated

if __name__ == "__main__":
    test_occupancy()
