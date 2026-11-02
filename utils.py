"""Funções reutilizáveis do projeto (usadas por todas as etapas).

Arquivos gerados (prefix = "qlearning-det", "sarsa-rainy" etc.):
    data/q-table-{prefix}-run{i}.csv   q-table da execução i
    results/rewards-{prefix}.npy       reward por episódio (runs x episódios)
    results/avaliacao-{prefix}.csv     métricas da avaliação gulosa
"""

import contextlib
import io

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from numpy import loadtxt

import config

CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # curva i usa CORES[i]
COR_META = "#0b0b0b"
COR_REFERENCIA = "#52514e"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"


def make_env(rainy=False):
    """Taxi-v4 sem TimeLimit (.env), determinístico ou chuvoso (p = 0.6)."""
    if rainy:
        return gym.make(config.ENV_ID, is_rainy=True,
                        rainy_probability=config.RAINY_PROBABILITY).env
    return gym.make(config.ENV_ID).env


def default_prefix(AgentClass, rainy):
    """Ex.: "qlearning-det", "sarsa-rainy"."""
    return f"{AgentClass.__name__.lower()}-{'rainy' if rainy else 'det'}"


def q_table_path(prefix, run):
    return config.DATA_DIR / f"q-table-{prefix}-run{run}.csv"


def rewards_path(prefix):
    return config.RESULTS_DIR / f"rewards-{prefix}.npy"


def run_experiments(AgentClass, rainy, n_runs=config.N_RUNS, prefix=None,
                    episodes=config.EPISODES):
    """Treina n_runs vezes (seeds de config.SEEDS) e salva q-tables e rewards.

    AgentClass precisa ter a interface de QLearning. Retorna a matriz de
    rewards (n_runs x episodes).
    """
    if n_runs > len(config.SEEDS):
        raise ValueError(f"n_runs={n_runs}, mas config.SEEDS só tem {len(config.SEEDS)} seeds.")
    if prefix is None:
        prefix = default_prefix(AgentClass, rainy)
    config.DATA_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    rewards = np.zeros((n_runs, episodes))
    for i in range(n_runs):
        env = make_env(rainy)
        agent = AgentClass(env, config.ALPHA, config.GAMMA, config.EPSILON,
                           config.EPSILON_MIN, config.EPSILON_DEC, episodes,
                           seed=config.SEEDS[i], max_steps=config.MAX_STEPS)
        with contextlib.redirect_stdout(io.StringIO()):  # esconde o contador de train()
            agent.train(str(q_table_path(prefix, i)), None)
        env.close()
        rewards[i] = agent.rewards_per_episode
        print(f"{prefix} run{i} (seed {config.SEEDS[i]}): reward médio nos últimos "
              f"{config.JANELA_MEDIA_MOVEL} episódios = "
              f"{rewards[i, -config.JANELA_MEDIA_MOVEL:].mean():+.2f}")

    np.save(rewards_path(prefix), rewards)
    return rewards


def evaluate(q_table, env, n=config.N_EVAL, max_steps=config.MAX_STEPS, seed=None):
    """Roda a política gulosa por n episódios (padrão da aula).

    A seed vale só no 1º reset. Retorna listas de timesteps, rewards e sucessos.
    """
    timesteps, rewards, successes = [], [], []
    for i in range(n):
        (state, _) = env.reset(seed=seed if i == 0 else None)
        done = False
        epochs, total_reward = 0, 0
        while (not done) and (epochs < max_steps):
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
            epochs += 1
            total_reward += reward
        timesteps.append(epochs)
        rewards.append(total_reward)
        successes.append(bool(done))
    return timesteps, rewards, successes


def print_evaluation(timesteps, rewards, successes):
    """Prints no formato da aula."""
    print("Average actions taken: {:.2f}".format(np.mean(timesteps)))
    print("Standard deviation actions taken: {:.2f}".format(np.std(timesteps)))
    print("Average reward: {:.2f}".format(np.mean(rewards)))
    print("Standard deviation reward: {:.2f}".format(np.std(rewards)))
    print("Success rate: {:.0%}".format(np.mean(successes)))


def evaluate_runs(prefix, rainy, n_runs=config.N_RUNS, n=config.N_EVAL):
    """Avalia as q-tables salvas (loadtxt) e salva results/avaliacao-{prefix}.csv.

    Retorna (dict de matrizes n_runs x n, DataFrame).
    """
    resultados = {"timesteps": [], "rewards": [], "sucesso": []}
    for i in range(n_runs):
        q_table = loadtxt(q_table_path(prefix, i), delimiter=",")
        env = make_env(rainy)
        timesteps, rewards, successes = evaluate(q_table, env, n=n, seed=config.SEED_EVAL)
        env.close()
        print(f"--- {prefix} run{i} ({n} episódios)")
        print_evaluation(timesteps, rewards, successes)
        resultados["timesteps"].append(timesteps)
        resultados["rewards"].append(rewards)
        resultados["sucesso"].append(successes)
    resultados = {k: np.array(v) for k, v in resultados.items()}

    def resumo(ts, rw, ok):
        return {"timesteps_media": ts.mean(), "timesteps_dp": ts.std(),
                "reward_media": rw.mean(), "reward_dp": rw.std(), "taxa_sucesso": ok.mean()}

    linhas = {f"run{i}": resumo(*(resultados[k][i] for k in resultados)) for i in range(n_runs)}
    linhas["todas"] = resumo(*(resultados[k].ravel() for k in resultados))
    tabela = pd.DataFrame(linhas).T
    tabela.index.name = "q-table"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    tabela.to_csv(config.RESULTS_DIR / f"avaliacao-{prefix}.csv")
    return resultados, tabela


def moving_average(x, janela):
    """Média móvel no último eixo (só janelas completas)."""
    c = np.cumsum(x, axis=-1)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), c], axis=-1)
    return (c[..., janela:] - c[..., :-janela]) / janela


def _estilo(ax):
    ax.grid(True, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_EIXO)
    ax.tick_params(colors="#52514e")


def _fmt(valor):
    return f"{valor:+.2f}".replace(".", ",")


def plot_learning_curves(dict_curvas, meta, titulo, arquivo, referencias=None,
                         janela=config.JANELA_MEDIA_MOVEL):
    """Curvas de aprendizado com a meta, salvas em results/arquivo.

    dict_curvas: {nome: matriz runs x episódios}, até 4 curvas.
    referencias: {rótulo: valor}, linhas horizontais extras (opcional).
    Painel de cima: visão geral; de baixo: zoom perto da meta. Retorna a Figure.
    """
    referencias = referencias or {}
    if len(dict_curvas) > len(CORES):
        raise ValueError(f"No máximo {len(CORES)} curvas por gráfico.")

    fig, (ax_geral, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True,
                                            layout="constrained")
    handles, rotulos = [], []
    valores_zoom = [meta, *referencias.values()]

    for cor, (nome, rewards) in zip(CORES, dict_curvas.items()):
        rewards = np.atleast_2d(np.asarray(rewards, dtype=float))
        n_runs, n_ep = rewards.shape
        media = rewards.mean(axis=0)
        suave = moving_average(rewards, janela)
        suave_media, suave_dp = suave.mean(axis=0), suave.std(axis=0)
        x = np.arange(1, n_ep + 1)
        x_suave = x[janela - 1:]

        fina, = ax_geral.plot(x, media, color=cor, lw=0.6, alpha=0.3)  # no zoom vira ruído
        for ax in (ax_geral, ax_zoom):
            faixa = ax.fill_between(x_suave, suave_media - suave_dp, suave_media + suave_dp,
                                    color=cor, alpha=0.22, lw=0)
            grossa, = ax.plot(x_suave, suave_media, color=cor, lw=2)
        handles += [fina, (faixa, grossa)]
        rotulos += [f"{nome}: média das {n_runs} execuções, por episódio",
                    f"{nome}: média móvel de {janela} episódios ± 1 desvio padrão entre execuções"]

        metade = suave_media[len(suave_media) // 2:]
        valores_zoom += [(metade - suave_dp[len(suave_dp) // 2:]).min(),
                         (metade + suave_dp[len(suave_dp) // 2:]).max()]

    linhas = [(f"Meta: {_fmt(meta)}", meta, COR_META, "--")]
    linhas += [(rotulo, valor, COR_REFERENCIA, estilo)
               for (rotulo, valor), estilo in zip(referencias.items(), [":", "-."] * 4)]
    for rotulo, valor, cor, estilo in linhas:
        for ax in (ax_geral, ax_zoom):
            linha = ax.axhline(valor, color=cor, ls=estilo, lw=1.5)
        ax_zoom.annotate(rotulo, xy=(0, valor), xycoords=("axes fraction", "data"),
                         xytext=(6, 3), textcoords="offset points",
                         ha="left", va="bottom", fontsize=9, color=cor)
        handles.append(linha)
        rotulos.append(rotulo)

    baixo, alto = min(valores_zoom), max(valores_zoom)
    folga = max(0.25 * (alto - baixo), 2.0)
    ax_zoom.set_ylim(baixo - folga, alto + folga)

    ax_geral.set_title("Visão geral", loc="left", fontsize=11)
    ax_zoom.set_title("Zoom no eixo y: patamar final e meta", loc="left", fontsize=11)
    for ax in (ax_geral, ax_zoom):
        _estilo(ax)
        ax.set_ylabel("Reward acumulado no episódio")
    ax_zoom.set_xlabel("Episódio de treino")
    ax_zoom.set_xlim(1, n_ep)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.legend(handles, rotulos, loc="outside lower center", ncol=1, frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=1)}, fontsize=9)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(config.RESULTS_DIR / arquivo, dpi=150, bbox_inches="tight")
    return fig


def _transicoes(env):
    """Modelo env.unwrapped.P em vetores: índice s*n_a+a, prob., s', r, terminal."""
    P = env.unwrapped.P
    n_a = env.action_space.n
    sa, prob, prox, rew, term = [], [], [], [], []
    for s, acoes in P.items():
        for a, saidas in acoes.items():
            for p, s2, r, t in saidas:
                sa.append(s * n_a + a)
                prob.append(p)
                prox.append(s2)
                rew.append(r)
                term.append(t)
    return (np.array(sa), np.array(prob), np.array(prox),
            np.array(rew, dtype=float), np.array(term, dtype=float))


def _backup(trans, v, n_s, n_a):
    """Q(s,a) = soma p * (r + V(s')), com V = 0 no terminal."""
    sa, prob, prox, rew, term = trans
    alvo = prob * (rew + (1 - term) * v[prox])
    return np.bincount(sa, weights=alvo, minlength=n_s * n_a).reshape(n_s, n_a)


def optimal_q_table(env, max_steps=config.MAX_STEPS):
    """Q* exata: iteração de valor sem desconto, episódios de até max_steps passos."""
    n_s, n_a = env.observation_space.n, env.action_space.n
    trans = _transicoes(env)
    q = np.zeros((n_s, n_a))
    for _ in range(max_steps):
        q = _backup(trans, q.max(axis=1), n_s, n_a)
    return q


def expected_reward(q_table, env, epsilon=0.0, max_steps=config.MAX_STEPS):
    """Reward médio esperado (exato) da política epsilon-greedy de q_table."""
    n_s, n_a = q_table.shape
    trans = _transicoes(env)
    pi = np.full((n_s, n_a), epsilon / n_a)
    pi[np.arange(n_s), np.argmax(q_table, axis=1)] += 1 - epsilon
    v = np.zeros(n_s)
    for _ in range(max_steps):
        v = (pi * _backup(trans, v, n_s, n_a)).sum(axis=1)
    return float(env.unwrapped.initial_state_distrib @ v)
