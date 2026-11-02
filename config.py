"""Configuração única do projeto. Não altere entre etapas."""

from pathlib import Path

# Ambiente
ENV_ID = "Taxi-v4"            # requer gymnasium >= 1.3
RAINY_PROBABILITY = 0.6
MAX_STEPS = 200

# Hiperparâmetros (sugestão da aula de Sarsa): epsilon fixo em 0.1
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
EPSILON_MIN = 0.1
EPSILON_DEC = 1
EPISODES = 5000
EPISODES_LONGO = 10000        # só no teste de convergência do Sarsa (item 3)

# Experimentos
N_RUNS = 5
SEEDS = [0, 1, 2, 3, 4]       # execução i usa SEEDS[i]
N_EVAL = 100
SEED_EVAL = 1000

# Meta (o ótimo exato é +7.93) e gráficos
META_REWARD = 7.0
JANELA_MEDIA_MOVEL = 100

RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"
RESULTS_DIR = RAIZ / "results"
