import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configuração de caminhos
sys.path.append(str(Path(__file__).parent.parent))
from core.occupancy_grid import OccupancyGrid

def simulate_environment(grid_size=(20, 20), n_obstacles=10):
    """Cria ambiente com obstáculos aleatórios"""
    grid = OccupancyGrid(*grid_size)
    obstacles = []
    
    # Garante obstáculos únicos
    while len(obstacles) < n_obstacles:
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        if (x, y) not in obstacles:
            obstacles.append((x, y))
    
    return grid, obstacles

def generate_movement_data(grid, obstacles, n_samples=100):
    """Gera dados de treino (X, y)"""
    # Inicializa como listas vazias
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        # 1. Simula posição aleatória do robô
        robot_x, robot_y = np.random.randint(0, grid.grid.shape[1]), np.random.randint(0, grid.grid.shape[0])
        
        # 2. Simula detecções do sensor
        sensor_readings = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for r in range(1, 5):
                x = int(robot_x + r * np.cos(angle))
                y = int(robot_y + r * np.sin(angle))
                
                if 0 <= x < grid.grid.shape[1] and 0 <= y < grid.grid.shape[0]:
                    occupied = (x, y) in obstacles
                    sensor_readings.append(((x, y), occupied))
                    if occupied: break
        
        # 3. Atualiza a grade
        grid.update(sensor_readings)
        
        # 4. Calcula movimento ideal (exemplo simplificado)
        if obstacles:
            closest = min(obstacles, key=lambda p: (p[0]-robot_x)**2 + (p[1]-robot_y)**2)
            dx = 0.5 if closest[0] < robot_x else -0.5
            dy = 0.5 if closest[1] < robot_y else -0.5
        else:
            dx, dy = np.random.uniform(-0.5, 0.5, size=2)
        
        # 5. Armazena os dados
        X_data.append(grid.grid.copy())
        y_data.append([dx, dy])
    
    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    # Configuração
    GRID_SIZE = (20, 20)
    N_OBSTACLES = 15
    N_SAMPLES = 100
    
    # Execução
    try:
        print("Iniciando simulação...")
        grid, obstacles = simulate_environment(GRID_SIZE, N_OBSTACLES)
        X, y = generate_movement_data(grid, obstacles, N_SAMPLES)
        
        # Verificação
        print(f"Shape de X: {X.shape}")  # Deve ser (100, 20, 20)
        print(f"Shape de y: {y.shape}")  # Deve ser (100, 2)
        
        # Salva os dados
        Path("training_data").mkdir(exist_ok=True)
        np.save("training_data/X.npy", X)
        np.save("training_data/y.npy", y)
        print("Dados salvos com sucesso em 'training_data/'")
        
        # Visualização de exemplo
        grid.visualize("Grade Final")
        
    except Exception as e:
        print(f"Erro durante a simulação: {str(e)}")