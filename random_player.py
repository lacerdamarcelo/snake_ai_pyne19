from snake import Snake
import random
import time

env = Snake()
while True:
	observables = env.reset()
	terminal = False
	while terminal is False:
		env.render()
		observables, reward, terminal, info = env.step(random.randint(0, env.actions_set_size-1))
		time.sleep(0.1)
