import time
import pickle
import random
import numpy as np
from snake import Snake


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
        '''
        Observáveis: [x_maçã - x_cabeça, y_maçã - y_cabeça, x_ponta_calda - x_cabeça, y_ponta_calda - y_cabeça]
        Esta função deve verificar se os observáveis representam ou não um novo estado.
        Cada estado é representado por uma das possíveis combinações de valores das variáveis
        que compõem os observáveis, sendo estes mapeados em um número inteiro (0, 1, 2, 3...).
        Se o estado existir, deve ser retornado o valor inteiro correspondente. Caso contrário,
        uma nova linha deve ser adicionada na matriz dos valores Q com todos os valores zerados e
        o número inteiro correspondente ao novo estado deve ser retornado.
    
        Pseudo-código:

        Se self.q_table não é None, faça:
            Se os observáveis não estiverem em self.states_list (não tiverem sido vistos antes):
                Acrescenta os observáveis na lista de observáveis (self.states_list);
                Acrescenta uma nova linha com 4 colunas zerada em self.q_table (no final da matriz);
                Retorna o tamanho do self.states_list menos 1;
            Caso contrário, faça:
                Retorna o índice do observável na lista self.states_list;
        Caso contrário, faça:
            Coloca o único estado definido pelos observables em self.states_list;
            Cria matriz Q com uma única linha zerada (4 colunas) e atribuir a self.q_table;
            Retorna 0 (zero);
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

        Pseudo-código:

        Sorteia-se um número aleatório entre 0 e 1 com distribuição uniforme;
        Se esse número for menor que epsilon:
            Escolhe uma das 4 ações aleatoriamente;
        Caso contrário, faça:
            Escolhe a ação de maior valor da matriz Q na linha correspondente ao estado state (índice da linha);
        Retorna ação escolhida;
        '''
        pass

    def update_q_table(self, current_state, current_action, current_reward, next_state):
        '''
        Os valores Q devem ser atualizados. Deve-se identificar a ação a(t+1) que maximiza Q(S(t+1),a(t+1)),
        sendo S(t+1) o próximo estado (next_state). Depois disso, o novo valor de Q(S,a) deve ser calculado,
        atualizando a matriz.

        Pseudo-código:

        Calcula o valor dq Q do current_state e current_action;
        Pega o valor máximo de Q do próximo estado (next_state): next_max;
        Calcula o próximo valor de Q do current_state e current_action através de: Q(a(t),S(t)) =
                                    (1-alpha) * Q(a(t),S(t)) + alpha * (reward + gamma * next_max)
        '''
        pass

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
