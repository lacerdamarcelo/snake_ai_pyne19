import time
from snake import Snake
from q_learning_player import QLearningPlayer

player = QLearningPlayer()
player.load_model('q_table.pkl', 'states_list.pkl')
env = Snake()
while True:
    observables = env.reset()
    state = player.infer_state(observables)
    terminal = False
    while terminal is False:
        env.render()
        state = player.infer_state(observables)
        action = player.choose_action(state)
        observables, reward, terminal, info = env.step(action)
        time.sleep(0.1)