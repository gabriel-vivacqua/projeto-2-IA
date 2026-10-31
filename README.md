# projeto-2-IA

Q-learning e Sarsa no `Taxi-v4` (Gymnasium) — Insper, Inteligência Artificial e Robótica.

```bash
pip install -r requirements.txt
jupyter notebook relatorio.ipynb
```

- `QLearning.py`: classe da aula, completa (a `Sarsa` deve ter a mesma interface)
- `utils.py`: ambiente, treino em 5 execuções, avaliação e gráficos
- `config.py`: hiperparâmetros únicos do projeto (não alterar)
- `relatorio.ipynb`: relatório
- `data/q-table-{prefix}-run{i}.csv`: q-tables
- `results/`: `rewards-{prefix}.npy`, `avaliacao-{prefix}.csv` e gráficos

`prefix` = `qlearning-det`, `sarsa-det`, `qlearning-rainy` ou `sarsa-rainy`.
