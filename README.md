# projeto-2-IA

Comparação de **Q-learning** e **Sarsa** no ambiente `Taxi-v4` (Gymnasium), nas versões determinística e com chuva. Projeto 2 de Inteligência Artificial e Robótica, Insper.

O relatório completo, com código, gráficos e discussão, está em [`relatorio.ipynb`](relatorio.ipynb).

## Integrantes

- Gabriel Vivacqua
- Rafael Santos
- Lucas Watanabe
- Joaquim Figueredo

## Como instalar e rodar

Testado com Python 3.11 e 3.12, gymnasium 1.4.0 e matplotlib 3.11.

```bash
pip install -r requirements.txt
jupyter notebook relatorio.ipynb
```

No Jupyter, use **Kernel > Restart & Run All**. As seeds são fixas, então a execução reproduz as q-tables e as métricas. Ela sobrescreve os arquivos de `data/` e `results/` e leva alguns minutos.

## Estrutura de pastas

```text
projeto-2-IA/
├── relatorio.ipynb    relatório: introdução, itens 1 a 5 e conclusões
├── QLearning.py       classe QLearning da aula, completa
├── Sarsa.py           classe Sarsa, com a mesma interface
├── utils.py           ambiente, treino em 5 execuções, avaliação e gráficos
├── config.py          hiperparâmetros, seeds e meta (únicos para todo o projeto)
├── requirements.txt
├── data/              q-tables: q-table-{prefix}-run{i}.csv
└── results/
    ├── rewards-{prefix}.npy            reward por episódio (5 × 5000)
    ├── avaliacao-{prefix}.csv          métricas de avaliação por q-table
    ├── avaliacao-item4-{algoritmo}-{det|rainy}.npz   episódios do item 4
    ├── avaliacao-item4.csv             resumo do item 4
    ├── reward-exato-cenarios.csv       reward esperado exato por cenário
    ├── resumo-final.csv                tabela-resumo final
    └── *.png                           gráficos
```

`prefix` é `qlearning-det`, `sarsa-det`, `qlearning-rainy` ou `sarsa-rainy` e indica o ambiente de **treino**. Há também `sarsa-det-longo`, o Sarsa com 10.000 episódios do teste de convergência do item 3.

Nos `.npz` do item 4, `det` e `rainy` indicam o ambiente de **teste**; as q-tables são sempre as treinadas no determinístico.

## Configuração

| Parâmetro | Valor |
|---|---|
| α | 0,1 |
| γ | 0,99 |
| ε | 0,1 (fixo) |
| Episódios de treino | 5000 |
| Limite de passos | 200 |
| Execuções | 5 (seeds 0 a 4) |
| Avaliação | 100 episódios por q-table, sem exploração |
| Chuva | `rainy_probability = 0.6` |
| Meta | reward médio ≥ +7,0 no ambiente determinístico (o ótimo é +7,93) |

## Resumo dos resultados

Avaliação com a política gulosa, em 500 episódios por cenário. A última coluna é o reward esperado exato, calculado sobre os 300 estados iniciais e todas as transições, e é a medida mais confiável para comparar os cenários.

| Algoritmo | Treino | Teste | Timesteps (média ± desvio) | Sucesso | Reward médio | Reward exato |
|---|---|---|---|---:|---:|---:|
| Q-learning | Determinístico | Determinístico | 12,94 ± 2,59 | 100,0 % | +8,06 | +7,93 |
| Sarsa | Determinístico | Determinístico | 13,26 ± 8,76 | 99,8 % | +7,70 | +6,84 |
| Q-learning | Determinístico | Com chuva | 25,58 ± 14,12 | 99,6 % | −4,67 | −6,64 |
| Sarsa | Determinístico | Com chuva | 28,11 ± 24,70 | 98,2 % | −7,49 | −8,90 |
| Q-learning | Com chuva | Com chuva | 26,34 ± 17,92 | 99,2 % | −5,51 | −7,33 |
| Sarsa | Com chuva | Com chuva | 56,24 ± 68,09 | 82,0 % | −39,02 | −55,25 |

- No ambiente determinístico, o Q-learning superou a meta nas cinco execuções e ficou praticamente na política ótima. O Sarsa ficou próximo, mas duas de suas cinco q-tables ficam abaixo de +7,0 no cálculo exato. Com 10.000 episódios, os ciclos da política gulosa somem e ele chega a +7,93, então é falta de treino.
- Com chuva, a meta de +7,0 não é alcançável: a política ótima rende −3,96.
- Treinar diretamente com chuva não gerou política mais robusta com 5000 episódios. O Q-learning empatou com a q-table determinística, e o Sarsa piorou muito, com 18 % dos episódios presos até o limite de passos.
