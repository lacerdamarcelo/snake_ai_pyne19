import pickle
import random
import numpy as np


class QLearningPlayer:
       
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1): 
        # Hiperparâmetros
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Inicializando matriz Q e lista de estados.
        self.states_list = []
        self.q_table = None

    def infer_state(self, observables_list):
        observables = str(observables_list)
        if self.q_table is not None:
            if observables not in self.states_list:
                self.states_list.append(observables)
                self.q_table = np.append(self.q_table, [np.zeros(4)], axis=0)
                return len(self.states_list) - 1
            else:
                return self.states_list.index(observables)
        else:
            self.states_list = [observables]
            self.q_table = np.zeros([1, 4])
            return 0

    '''
    Ações possíveis:
    0 - cima;
    1 - direita;
    2 - baixo;
    3 - esquerda.
    '''
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

    def save_model(self, q_table_path, states_list_path):
        with open(q_table_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        with open(states_list_path, 'wb') as f:
            pickle.dump(self.states_list, f)

    def load_model(self, q_table_path, states_list_path):
        with open(q_table_path, 'rb') as f:
            self.q_table = pickle.load(f)
        with open(states_list_path, 'rb') as f:
            self.states_list = pickle.load(f)
