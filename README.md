# Experimento de Espuma Probabilística em Tempo Real

## 📌 Visão Geral
Simulação comparativa de modelos de ocupação espacial, com foco em:
- Espuma Probabilística
- Grade de Ocupação Binária
- Campos Potenciais

## 🛠️ Configuração do Ambiente
```bash
# Dependências
pip install numpy numba matplotlib psutil

# Para visualização 3D (opcional)
pip install mayavi pyqt5
```

## 🗂️ Estrutura de Arquivos
```
experimento_espuma/
├── data/
│   ├── ground_truth.npy    # Mapa de referência (128x128)
│   └── trajectories.npy    # Trajetórias de 3 objetos móveis
├── core/                   # Implementações dos algoritmos
├── scripts/                # Scripts auxiliares
└── visualize.py            # Visualização principal
```

## ▶️ Como Executar
```bash
# Visualização básica
python visualize.py

# Benchmark completo
python scripts/benchmark.py

# Gerar novos dados de teste
python scripts/generate_data.py
```

## 📊 Métricas Coletadas
| Métrica               | Descrição                          |
|-----------------------|-----------------------------------|
| Latência              | Tempo por iteração (ms)           |
| Acurácia (MSE)        | Erro vs. ground truth             |
| Uso de CPU            | Percentual de utilização          |
| Detecção de Móveis    | % de obstáculos móveis detectados |

## 🤝 Contribuição
1. Clone o repositório
2. Crie uma branch: `git checkout -b minha-feature`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin minha-feature`

## 📜 Licença
MIT