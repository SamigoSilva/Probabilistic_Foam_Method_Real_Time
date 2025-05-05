import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoamDataset(Dataset):
    """Dataset para treino da rede neural"""
    def __init__(self, grid_data: np.ndarray, targets: np.ndarray):
        """
        Args:
            grid_data: Array numpy com formatos [n_samples, height, width]
            targets: Array numpy com os targets correspondentes
        """
        self.grid_data = torch.FloatTensor(grid_data).unsqueeze(1)  # Adiciona channel dim
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.grid_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.grid_data[idx], self.targets[idx]

class FoamPredictor(nn.Module):
    """Arquitetura leve para predição em tempo real"""
    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()
        h, w = input_shape
        
        self.model = nn.Sequential(
            # Camada de entrada
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # [batch, 8, h, w]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 8, h//2, w//2]
            
            # Camada intermediária
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # [batch, 16, h//2, w//2]
            nn.ReLU(),
            
            # Saída
            nn.Flatten(),
            nn.Linear(16 * (h//2) * (w//2), 1)  # Ajustar conforme necessidade
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class NeuralFoamPredictor:
    """Wrapper para o modelo neural com otimizações"""
    def __init__(self, config: Dict):
        self.device = self._get_device()
        self.model = FoamPredictor((config['grid']['height'], config['grid']['width'])).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        # Congela camadas se necessário
        if config.get('freeze', False):
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _get_device(self) -> torch.device:
        """Seleciona dispositivo automaticamente com fallback para CPU"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, train_loader: DataLoader, epochs: int = 10) -> Dict[str, float]:
        """Loop de treino otimizado"""
        self.model.train()
        metrics = {'loss': []}
        
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
            metrics['loss'].append(loss.item())
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        return metrics
    
    def predict(self, grid: np.ndarray) -> np.ndarray:
        """Predição otimizada para tempo real"""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self.device)
            output = self.model(inputs)
        return output.cpu().numpy()
    
    def save(self, path: str):
        """Exporta o modelo otimizado"""
        torch.jit.script(self.model).save(path)
        logger.info(f"Model saved to {path}")

# Exemplo de uso (teste mínimo)
if __name__ == "__main__":
    config = {
        'grid': {'height': 80, 'width': 80},
        'freeze': False
    }
    
    # Dummy data
    dummy_data = np.random.rand(100, 80, 80)
    dummy_targets = np.random.rand(100, 1)
    
    # Pipeline completo
    predictor = NeuralFoamPredictor(config)
    dataset = FoamDataset(dummy_data, dummy_targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Treino (opcional)
    predictor.train(loader, epochs=5)
    
    # Predição
    test_grid = np.random.rand(80, 80)
    prediction = predictor.predict(test_grid)
    print(f"Prediction: {prediction}")
