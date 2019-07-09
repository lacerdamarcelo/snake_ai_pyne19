import time
from snake import Snake
from q_learning_player import QLearningPlayer

print('Training model...')
player = QLearningPlayer()
player.load_model('q_table.pkl', 'states_list.pkl')
env = Snake()
epoch = 10000
while True:
    observables = env.reset()
    state = player.infer_state(observables)
    terminal = False
    while terminal is False:
        #env.render()
        action = player.choose_action(state)
        observables, reward, terminal, info = env.step(action)
        next_state = player.infer_state(observables)
        player.update_q_table(state, action, reward, next_state)
        state = next_state
    epoch += 1
    if epoch % 100 == 0:
        print('Saving checkpoint at epoch: %d' % epoch)
        print('Hi Score: %d' % env.max_score)
        player.save_model('q_table.pkl', 'states_list.pkl')
