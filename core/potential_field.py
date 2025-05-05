import ctypes
import os
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from numba import njit

def load_dll() -> ctypes.CDLL:
    """Carrega a DLL com verificações de segurança"""
    dll_path = Path(__file__).parent / 'potential_field.dll'
    print(f"Tentando carregar DLL de: {dll_path}")

    if not dll_path.exists():
        available_files = [f.name for f in Path(__file__).parent.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Arquivo DLL não encontrado em: {dll_path}\n"
            f"Arquivos disponíveis: {available_files}"
        )

    try:
        os.environ['PATH'] = str(dll_path.parent) + os.pathsep + os.environ.get('PATH', '')
        os.add_dll_directory(str(dll_path.parent))
        lib = ctypes.CDLL(str(dll_path))
        print("DLL carregada com sucesso!")
        return lib
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar DLL: {str(e)}")

lib = load_dll()

# Configuração dos tipos da função C++
lib.update_potential_field.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # grid
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.POINTER(ctypes.c_float),  # obstacles
    ctypes.c_int,                    # num_obstacles
    ctypes.c_float,                  # robot_x
    ctypes.c_float                   # robot_y
]
lib.update_potential_field.restype = None

@njit(fastmath=True)
def update_potential_field(
    grid: Union[List[List[float]], np.ndarray],
    obstacles: List[Tuple[float, float]],
    width: int = None,
    height: int = None,
    robot_pos: Tuple[float, float] = None
) -> List[List[float]]:
    """
    Wrapper Python para a função C++ de campo potencial
    
    Args:
        grid: Matriz 2D (lista de listas ou numpy array)
        obstacles: Lista de tuplas [(x1,y1), (x2,y2), ...]
        width: Largura do grid (opcional)
        height: Altura do grid (opcional)
        robot_pos: Posição (x,y) do robô (opcional, centro por padrão)
    
    Returns:
        Matriz atualizada com o campo potencial
    """
    # Verificação e conversão do grid
    if isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            raise ValueError("Numpy array deve ser 2D")
        height, width = grid.shape
        # Converte para array C compatível
        grid_flat = grid.astype(np.float32).flatten()
        # Cria ponteiro para os dados
        grid_ptr = grid_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    elif isinstance(grid, list):
        height = height or len(grid)
        width = width or (len(grid[0]) if height > 0 else 0)
        
        if len(grid) != height or any(len(row) != width for row in grid):
            raise ValueError("Dimensões do grid inconsistentes")
        
        # Cria array C e copia os dados
        grid_flat = (ctypes.c_float * (width * height))()
        for i in range(height):
            for j in range(width):
                grid_flat[i * width + j] = float(grid[i][j])
        grid_ptr = grid_flat
    else:
        raise TypeError("Grid deve ser lista de listas ou numpy array")

    # Processamento dos obstáculos
    obstacles_array = (ctypes.c_float * (2 * len(obstacles)))()
    for i, (x, y) in enumerate(obstacles):
        obstacles_array[2*i] = float(x)
        obstacles_array[2*i + 1] = float(y)

    # Posição do robô (padrão: centro)
    robot_x, robot_y = robot_pos if robot_pos else (width / 2.0, height / 2.0)

    # Chamada para a função C++
    lib.update_potential_field(
        grid_ptr, width, height,
        obstacles_array, len(obstacles),
        ctypes.c_float(robot_x), ctypes.c_float(robot_y)
    )

    # Converte o resultado de volta
    if isinstance(grid, np.ndarray):
        return grid_flat.reshape((height, width)).tolist()
    else:
        return [[grid_flat[i * width + j] for j in range(width)] for i in range(height)]
