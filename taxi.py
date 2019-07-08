import gym
import os

env = gym.make('Taxi-v2')
reward = 0
done = False
ob = None
_ = None
total_reward = 0
while True:
	env.reset()
	done = False
	while done is False:
		os.system('cls' if os.name == 'nt' else 'clear')
		env.render()
		print('Pontuação: ' + str(total_reward))
		action = int(input('Ação: '))
		ob, reward, done, _ = env.step(action)
		total_reward += reward