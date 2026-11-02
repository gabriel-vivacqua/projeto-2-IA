import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt

# Sarsa da aula, com a mesma interface de QLearning.

class Sarsa:

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
        self.max_steps = max_steps
        self.rewards_per_episode = []
        if seed is not None:
            random.seed(seed)
            self.env.action_space.seed(seed)
            self.env.reset(seed=seed)

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def train(self, filename, plotFile):
        actions_per_episode = []
        self.rewards_per_episode = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            action = self.select_action(state)
            rewards = 0
            done = False
            actions = 0

            while not done:
                next_state, reward, done, truncated, _ = self.env.step(action)

                old_value = self.q_table[state, action]
                if done:
                    next_value = 0
                else:
                    next_action = self.select_action(next_state)
                    next_value = self.q_table[next_state, next_action]
                new_value = old_value + self.alpha * (
                    reward + self.gamma * next_value - old_value
                )
                self.q_table[state, action] = new_value

                state = next_state
                if not done:
                    action = next_action
                actions = actions + 1
                rewards = rewards + reward

                if truncated or (self.max_steps is not None and actions >= self.max_steps):
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
