from snake import Snake
import numpy as np
import random
import time


class QLearningPlayer:
       
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1): 
        self.states_set = set([])
        self.states_list = []
        self.q_table = np.zeros([1, 4])

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def infer_state(self, observables_list):
        observables = str(observables_list)
        if observables not in self.states_set:
            self.states_set.add(observables)
            self.states_list.append(observables)
            self.q_table = np.append(self.q_table, [np.zeros(4)], axis=0)
            return len(self.states_list) - 1
        else:
            return self.states_list.index(observables)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = random.choice(np.argwhere(self.q_table[state] == np.amax(self.q_table[state])).flatten())
        return action

    def update_q_table(self, current_state, current_action, current_reward, next_state):
        old_value = self.q_table[current_state, current_action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * \
                    (current_reward + self.gamma * next_max)
        self.q_table[current_state, current_action] = new_value


player = QLearningPlayer()
env = Snake()
while True:
    observables = env.reset()
    state = player.infer_state(observables)
    terminal = False
    while terminal is False:
        env.render()
        action = player.choose_action(state)
        observables, reward, terminal, info = env.step(action)
        next_state = player.infer_state(observables)
        player.update_q_table(state, action, reward, next_state)
        state = next_state
