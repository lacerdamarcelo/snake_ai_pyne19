import random
import time
import numpy as np
from snake import Snake


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
        '''
        Esta função deve verificar se os observáveis representam ou não um novo estado.
        Cada estado é representado por uma das possíveis combinações de valores das variáveis
        que compõem os observáveis, sendo estes mapeados em um número inteiro (0, 1, 2, 3...).
        Se o estado existir, deve ser retornado o valor inteiro correspondente. Caso contrário,
        uma nova linha deve ser adicionada na matriz dos valores Q com todos os valores zerados e
        o número inteiro correspondente ao novo estado deve ser retornado. 
        '''
        pass

    '''
    Ações possíveis:
    0 - cima;
    1 - direita;
    2 - baixo;
    3 - esquerda.
    '''
    def choose_action(self, state):
        '''
        Esta função deve retornar a ação a ser tomada, dado o estado recebido como parâmetro.
        Deve-se sortear um número aleatório (distribuição uniforme). Se este for menor do que
        epsilon, então a escolha da ação deve ser aleatória. Caso contrário, deve-se escolher
        a ação 'a' que maximize o valor Q(S,a), sendo 'S' o estado corrente.
        '''
        pass

    def update_q_table(self, current_state, current_action, current_reward, next_state):
        '''
        Os valores Q devem ser atualizados. Deve-se identificar a ação a(t+1) que maximiza Q(S(t+1),a(t+1)),
        sendo S(t+1) o próximo estado (next_state) após a ação tomada "current_action". Depois disso,
        o novo valor de Q(S,a) deve ser calculado, atualizando a matriz.
        '''
        pass


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