import numpy as np
import yaml
from typing import Dict, List
from collections import deque

class Foam:
    def __init__(self, config_file="config/params.yaml"):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['grid']['width']
        self.height = self.config['grid']['height']
        self.grid = np.zeros((self.height, self.width))
        self.obstacles = self._generate_obstacles(n_obstacles, width, height)
        self.robot_pos = (width // 2, height // 2)
        self.history = deque(maxlen=30)  # Armazena apenas os últimos 30 frames
        
    def _generate_obstacles(self, n: int, w: int, h: int) -> List[tuple]:
        return [(np.random.randint(0, w), np.random.randint(0, h)) for _ in range(n)]
    
    def update(self, method: str, **kwargs):
        """Atualiza o grid conforme o método selecionado"""
        self.history.append(self.grid.copy())
        
        if method == "potential":
            from .potential_field import update_potential_field
            update_potential_field(self.grid, self.obstacles, *self.robot_pos)
        
        elif method == "occupancy":
            from .occupancy_grid import update_occupancy
            self.grid = update_occupancy(self.grid, kwargs.get('sensor_data'))
        
        elif method == "neural":
            from .neural_network import predict_next_step
            movement = predict_next_step(self.grid)
            self._apply_movement(movement)
    
    def get_metrics(self) -> Dict[str, float]:
        from .metrics import calculate_metrics
        return calculate_metrics(self.grid)
