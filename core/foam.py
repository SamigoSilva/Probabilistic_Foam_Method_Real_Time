# core/foam.py
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import logging
from .config_manager import ConfigManager
from .potential_field import update_potential_field
from .occupancy_grid import update_occupancy
from .neural_network import NeuralFoamPredictor

class Foam:
    """Classe principal da simulação de espuma probabilística"""
    
    def __init__(self, config_file: str = "config/params.yaml"):
        # Configuração inicial
        self.config = ConfigManager(config_file)
        self._setup_logging()
        self._init_grid()
        self._init_components()
        self.logger.info("Simulação inicializada com sucesso")

    def _setup_logging(self):
        """Configura sistema de logs"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_grid(self):
        """Inicializa grid com base nas configurações"""
        self.width = self.config.get_config("grid.width")
        self.height = self.config.get_config("grid.height")
        
        # Otimização: usar float32 e array pré-alocado
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.history = deque(maxlen=30)  # Buffer circular para histórico
        
        # Gera obstáculos uma única vez
        self.obstacles = self._generate_obstacles(
            n=20,  # Número fixo ou poderia vir do config
            width=self.width,
            height=self.height
        )
        
        # Posição inicial do robô no centro
        self.robot_pos = (self.width // 2, self.height // 2)

    def _init_components(self):
        """Inicializa componentes dinâmicos"""
        method = self.config.get_config("methods.default")
        
        if method == "neural":
            self.nn_predictor = NeuralFoamPredictor(self.config.config)
            self.logger.info("Modelo neural carregado")

    def _generate_obstacles(self, n: int, width: int, height: int) -> List[tuple]:
        """Gera obstáculos aleatórios otimizado"""
        return [
            (np.random.randint(0, width), 
            np.random.randint(0, height)
            for _ in range(n)
        ]

    def update(self, sensor_data: Optional[np.ndarray] = None):
        """Atualiza o grid conforme o método selecionado"""
        method = self.config.get_config("methods.default")
        
        try:
            if method == "potential":
                update_potential_field(
                    self.grid, 
                    self.obstacles,
                    *self.robot_pos
                )
                
            elif method == "occupancy" and sensor_data is not None:
                self.grid = update_occupancy(
                    self.grid,
                    sensor_data
                )
                
            elif method == "neural":
                self.grid = self.nn_predictor.predict(self.grid)
                self.robot_pos = self._update_robot_position()
            
            # Registra histórico (com downsampling para economizar memória)
            self.history.append(self.grid[::2, ::2].copy())
            
        except Exception as e:
            self.logger.error(f"Erro na atualização: {e}")
            raise

    def _update_robot_position(self) -> tuple:
        """Atualiza posição do robô baseado no gradiente do grid"""
        grad_y, grad_x = np.gradient(self.grid)
        new_x = max(0, min(self.width-1, self.robot_pos[0] + int(grad_x.mean())))
        new_y = max(0, min(self.height-1, self.robot_pos[1] + int(grad_y.mean())))
        return (new_x, new_y)

    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas-chave da simulação"""
        return {
            "mean_prob": float(np.mean(self.grid)),
            "max_prob": float(np.max(self.grid)),
            "entropy": self._calculate_entropy(),
            "robot_x": self.robot_pos[0],
            "robot_y": self.robot_pos[1]
        }

    def _calculate_entropy(self) -> float:
        """Calcula entropia do grid de forma otimizada"""
        prob = self.grid.clip(1e-10, 1-1e-10)  # Evita log(0)
        return float(-np.sum(prob * np.log2(prob)))

    def reset(self):
        """Reinicia a simulação mantendo configurações"""
        self.grid.fill(0)
        self.history.clear()
        self.robot_pos = (self.width // 2, self.height // 2)
        self.logger.info("Simulação reiniciada")

# Exemplo de uso
if __name__ == "__main__":
    foam = Foam()
    
    for _ in range(100):
        foam.update()
        print(foam.get_metrics())
