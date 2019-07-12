from snake import Snake
import random
import time

env = Snake()
while True:
	observables = env.reset()
	terminal = False
	while terminal is False:
		env.render()
		action = random.randint(0, env.actions_set_size-1)
		observables, reward, terminal, info = env.step(action)
		time.sleep(0.1)
