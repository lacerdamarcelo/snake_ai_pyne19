import time
from snake import Snake
from q_learning_player_incomplete import QLearningPlayer

print('Training model...')
# Inicializar player.
env = Snake()
epoch = 0
while True:
    observables = env.reset()
    # Inferir estado após reset ((re)inicialização do ambiente)
    terminal = False
    while terminal is False:
        env.render()
        # Escolher ação
        observables, reward, terminal, info = env.step(action)
        # Inferir novo estado após step
        # Atualizar qualidade da ação tomada no estado anterior (antes do step)
    epoch += 1
    if epoch % 100 == 0:
        print('Saving checkpoint at epoch: %d' % epoch)
        print('Hi Score: %d' % env.max_score)
        player.save_model('q_table.pkl', 'states_list.pkl')
