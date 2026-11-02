import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt

# Q-learning da aula. Mudanças marcadas com "MUDANÇA".

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes,
                 seed=None, max_steps=None):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.max_steps = max_steps  # MUDANÇA: limite de passos (.env remove o TimeLimit)
        self.rewards_per_episode = []  # MUDANÇA: reward de cada episódio
        if seed is not None:  # MUDANÇA: reprodutibilidade
            random.seed(seed)
            self.env.action_space.seed(seed)
            self.env.reset(seed=seed)

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample()  # Explora
        return np.argmax(self.q_table[state])  # Usa a q-table

    def train(self, filename, plotFile):
        actions_per_episode = []
        self.rewards_per_episode = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                # MUDANÇA: Q(s,a) += alpha * [r + gamma * max Q(s',.) - Q(s,a)]
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state, action] = new_value

                state = next_state
                actions = actions + 1
                rewards = rewards + reward

                if truncated or (self.max_steps is not None and actions >= self.max_steps):  # MUDANÇA
                    break

            actions_per_episode.append(actions)
            self.rewards_per_episode.append(rewards)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) + '\r')
                sys.stdout.flush()

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if plotFile is not None:
            self.plotactions(plotFile, actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.figure()
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile + ".jpg")
        plt.close()
