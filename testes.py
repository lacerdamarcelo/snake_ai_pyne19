import numpy as np
from q_learning_player import QLearningPlayer

player = QLearningPlayer()

print('Testando infer_state...')
assert player.infer_state([1, 2, 3]) == 0
assert player.infer_state([3, 4, 5]) == 1
assert player.infer_state([6, 7, 8]) == 2
assert player.infer_state([1, 2, 3]) == 0
assert player.infer_state([3, 4, 5]) == 1
assert player.infer_state([6, 7, 8]) == 2
assert (player.q_table == np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])).all()
print('infer_state funciona perfeitamente!')

print('Testando choose_action...')
player = QLearningPlayer(epsilon=0.0)
player.infer_state([1, 2, 3])
player.infer_state([3, 4, 5])
player.infer_state([6, 7, 8])
player.infer_state([1, 2, 3])
player.infer_state([3, 4, 5])
player.infer_state([6, 7, 8])
player.q_table = np.array([[10, 20, 15, 10], [15, 10, 30, 20], [20, 5, 2, 40]])
assert player.choose_action(0) == 1
assert player.choose_action(1) == 2
assert player.choose_action(2) == 3
print('choose_action funciona perfeitamente!')

print('Testando update_q_table...')
current_state = player.infer_state([1, 2, 3])
current_action = player.choose_action(current_state)
next_state = player.infer_state([6, 7, 8])
player.update_q_table(current_state, current_action, 20, next_state)
assert (player.q_table == np.array([[10, 22, 15, 10], [15, 10, 30, 20], [20, 5, 2, 40]])).all()
print('update_q_table funciona perfeitamente!')

print('Todos os testes foram executados com sucesso!')

