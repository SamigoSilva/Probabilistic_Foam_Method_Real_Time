import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import List, Tuple
from sklearn.metrics import confusion_matrix

class OccupancyGrid:
    def __init__(self, width: int, height: int):
        """
        Inicializa uma grade de ocupação bayesiana.
        
        Parâmetros:
            width: Largura do grid (eixo x)
            height: Altura do grid (eixo y)
        """
        self.grid = np.full((height, width), 0.5)  # 0.5 = incerteza inicial
        self.prob_occupied = 0.9  # P(detecção|obstáculo)
        self.prob_free = 0.2      # P(detecção|livre)
        self.history = [self.grid.copy()]  # Para animações

    def update(self, sensor_data: List[Tuple[Tuple[int, int], bool]], 
               false_positive: float = 0.05, 
               false_negative: float = 0.1):
        """
        Atualiza o grid com dados de sensor, incluindo ruído.
        
        Parâmetros:
            sensor_data: Lista de [((x, y), occupied)]
            false_positive: Chance de falso positivo
            false_negative: Chance de falso negativo
        """
        for (x, y), occupied in sensor_data:
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
                # Adiciona ruído às leituras
                if np.random.random() < (false_positive if not occupied else false_negative):
                    occupied = not occupied
                
                # Atualização bayesiana
                prior = self.grid[y, x]
                likelihood = self.prob_occupied if occupied else self.prob_free
                posterior = (likelihood * prior) / (
                    likelihood * prior + (1 - likelihood) * (1 - prior)
		)
                self.grid[y, x] = posterior
        
        self.history.append(self.grid.copy())

    def get_metrics(self, ground_truth: List[Tuple[int, int]]) -> dict:
        """
        Calcula métricas de desempenho comparando com obstáculos reais.
        
        Retorna:
            Dicionário com 'acuracia', 'f1_score', 'cobertura'
        """
        y_true = np.zeros_like(self.grid)
        for x, y in ground_truth:
            y_true[y, x] = 1
        
        y_pred = (self.grid > 0.7).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
        
        return {
            'acuracia': (tp + tn) / (tp + tn + fp + fn),
            'f1_score': (2 * tp) / (2 * tp + fp + fn),
            'cobertura': np.mean(self.grid != 0.5)
        }

    def visualize(self, title: str = "Grade de Ocupação", save_path: str = None):
        """Visualiza o grid atual"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.grid, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(label="Probabilidade de Obstáculo")
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def animate_evolution(self, save_path: str = "evolucao.mp4"):
        """Gera animação da evolução histórica"""
        fig = plt.figure()
        im = plt.imshow(self.history[0], cmap="RdYlGn", animated=True)
        
        def update(frame):
            im.set_array(self.history[frame])
            return im,
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), blit=True, interval=200
        )
        ani.save(save_path, writer='ffmpeg')
        plt.close()


# Exemplo de uso (teste automático se executado diretamente)
if __name__ == "__main__":
    print("=== TESTE DA GRADE DE OCUPAÇÃO ===")
    
    # 1. Inicialização
    grid = OccupancyGrid(10, 10)
    obstaculos_reais = [(2, 3), (7, 1)]
    
    # 2. Simulação com dados de sensor
    dados_sensor = [
        ((2, 3), True),   # Detecção correta
        ((5, 5), False),  # Espaço livre
        ((7, 1), False)   # Falso negativo (erro)
    ]
    
    grid.update(dados_sensor)
    print("Métricas:", grid.get_metrics(obstaculos_reais))
    
    # 3. Visualização
    grid.visualize("Grade após primeira atualização")
    
    # 4. Animação (se houver histórico)
    if len(grid.history) > 1:
        grid.animate_evolution()