# core/config_manager.py
import yaml
import os
import psutil
from typing import Dict, Any
import logging

class ConfigManager:
    """Gerenciador dinâmico de configurações para otimização em tempo real"""
    
    def __init__(self, config_path: str = "config/params.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._adapt_to_hardware()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Arquivo de configuração não encontrado: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Erro no arquivo YAML: {e}")
            raise

    def _adapt_to_hardware(self):
        """Ajusta parâmetros automaticamente baseado no hardware"""
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        cores = psutil.cpu_count(logical=False)
        
        # Ajustes para hardware limitado (como seu i5-2400)
        if mem_gb < 16 or cores < 4:
            self.logger.info("Hardware limitado detectado. Ajustando parâmetros...")
            self.config["grid"]["width"] = min(80, self.config["grid"]["width"])
            self.config["grid"]["height"] = min(80, self.config["grid"]["height"])
            self.config["simulation"]["max_steps"] = 500
            
            if "neural" in self.config["methods"]:
                self.config["methods"]["neural"]["batch_size"] = 4

    def get_config(self, key: str = None) -> Any:
        """Obtém valor de configuração com fallback seguro"""
        keys = key.split(".") if key else []
        val = self.config
        
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            self.logger.warning(f"Chave de configuração não encontrada: {key}")
            return None

    def update_config(self, key: str, value: Any):
        """Atualiza configuração dinamicamente"""
        keys = key.split(".")
        conf = self.config
        
        for k in keys[:-1]:
            conf = conf.setdefault(k, {})
        conf[keys[-1]] = value
        
        self.logger.info(f"Configuração atualizada: {key} = {value}")

    def save_config(self, path: str = None):
        """Salva configurações em disco"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f)
        self.logger.info(f"Configurações salvas em: {save_path}")

# Exemplo de uso mínimo
if __name__ == "__main__":
    # Cria config/params.yaml se não existir
    if not os.path.exists("config/params.yaml"):
        os.makedirs("config", exist_ok=True)
        base_config = {
            "grid": {
                "width": 100,
                "height": 100,
                "resolution": 0.1
            },
            "simulation": {
                "max_steps": 1000,
                "real_time": True
            },
            "methods": {
                "default": "potential",
                "neural": {
                    "batch_size": 8,
                    "model_path": "data/models/best_model.pth"
                }
            }
        }
        with open("config/params.yaml", 'w') as f:
            yaml.safe_dump(base_config, f)

    # Testa o gerenciador
    config = ConfigManager()
    print("Largura do grid:", config.get_config("grid.width"))
    config.update_config("grid.width", 80)
    config.save_config()
