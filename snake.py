import random
import os
import numpy as np
from gym.spaces import discrete, box

class Snake:
	def __init__(self, init_len=3, use_middle_point=False, board_size=(15, 15)):
		snake_initial_position = [random.randint(0, board_size[0]),
								  random.randint(0, board_size[1])]
		self.snake = [snake_initial_position]
		for i in range(1, 3):
			self.snake.append([self.snake[-1][0], self.snake[-1][1] + 1])
		self.board_size = board_size
		self.apple_position = [random.randint(0, board_size[0]),
							   random.randint(0, board_size[1])]
		while self.apple_position in self.snake:
			self.apple_position = [random.randint(0, board_size[0]),
								   random.randint(0, board_size[1])]
		self.score = 0
		self.last_deleted = self.snake[-1]
		self.max_score = 0
		self.time_alive = 0
		self.max_time_alive = 0
		self.use_middle_point = use_middle_point
		self.actions_set_size = 4


	def warp_element(self, element):
		if element[0] == self.board_size[0] + 1:
			element[0] = 0
		elif element[0] == -1:
			element[0] = self.board_size[0]
		if element[1] == self.board_size[1] + 1:
			element[1] = 0
		elif element[1] == -1:
			element[1] = self.board_size[1]
		return element

	def warp(self):
		for i in range(0, len(self.snake)):
			self.snake[i] = self.warp_element(self.snake[i])

	def check_collision(self):
		if self.snake[0] == self.apple_position:
			self.snake.append(self.last_deleted)
			while self.apple_position in self.snake:
				self.apple_position = [random.randint(0, self.board_size[0]),
									   random.randint(0, self.board_size[1])]
			self.score += 1
			return 1, False
		elif self.snake[0] in self.snake[1:]:
			return -1, True
		else:
			return -0.1, False
	
	def reset(self):
		self.max_score = self.score if self.score > self.max_score else self.max_score
		self.max_time_alive = self.time_alive if self.time_alive > self.max_time_alive else self.max_time_alive
		snake_initial_position = [random.randint(0, self.board_size[0]),
								  random.randint(0, self.board_size[1])]
		self.snake = [snake_initial_position]
		for i in range(1, 3):
			self.snake.append([self.snake[-1][0], self.snake[-1][1] + 1])
		self.apple_position = [random.randint(0, self.board_size[0]),
							   random.randint(0, self.board_size[1])]
		while self.apple_position in self.snake:
			self.apple_position = [random.randint(0, self.board_size[0]),
								   random.randint(0, self.board_size[1])]
		self.score = 0
		self.time_alive = 0
		if self.use_middle_point:
			observables = self.calculate_observables_half()
		else:
			observables = self.calculate_observables()
		return observables

	# up: 0
	# right: 1
	# down: 2
	# left: 3
	def apply_action(self, action):
		if action == 0:
			new_head_position = [self.snake[0][0] - 1, self.snake[0][1]]
		elif action == 1:
			new_head_position = [self.snake[0][0], self.snake[0][1] + 1]
		elif action == 2:
			new_head_position = [self.snake[0][0] + 1, self.snake[0][1]]
		else:
			new_head_position = [self.snake[0][0], self.snake[0][1] - 1]
		return new_head_position

	def calculate_observables(self):
		observables = [self.apple_position[0] - self.snake[0][0],
					   self.apple_position[1] - self.snake[0][1]]
		observables.append(self.snake[-1][0] - self.snake[0][0])
		observables.append(self.snake[-1][1] - self.snake[0][1])
		return observables

	def calculate_observables_half(self):
		observables = [self.apple_position[0] - self.snake[0][0],
					   self.apple_position[1] - self.snake[0][1],
					   self.snake[int(len(self.snake) / 2)][0] - self.snake[0][0],
					   self.snake[int(len(self.snake) / 2)][1] - self.snake[0][1],
					   self.snake[-1][0] - self.snake[0][0],
					   self.snake[-1][1] - self.snake[0][1]]
		return observables

	def render(self):
		os.system('cls' if os.name == 'nt' else 'clear')
		print("Score: " + str(self.score))
		print("Hi Score: " + str(self.max_score))
		print("Time alive: " + str(self.time_alive))
		print("Hi Time alive: " + str(self.max_time_alive))
		for i in range(-1, self.board_size[0] + 2):
			if i != -1:
				print('')
			for j in range(-1, self.board_size[1] + 2):
				if [i, j] == self.apple_position:
					print('a', end='')
				elif [i, j] in self.snake:
					if self.snake.index([i, j]) == 0:
						print('c', end='')
					else:
						print('s', end='')
				elif i == -1 or i == self.board_size[0] + 1 or j == -1 or j == self.board_size[1] + 1:
					print('#', end='')
				else:
					print(' ', end='')
		print("\n")

	def step(self, action):
		self.warp()
		self.time_alive += 1
		new_head_position = self.apply_action(action)
		new_head_position = self.warp_element(new_head_position)
		if new_head_position != self.snake[1]:
			self.snake.insert(0, new_head_position)
		else:
			if action == 0:
				action = 2
			elif action == 1:
				action = 3
			elif action == 2:
				action = 0
			else:
				action = 1
			new_head_position = self.apply_action(action)
			new_head_position = self.warp_element(new_head_position)
			self.snake.insert(0, new_head_position)
		self.last_deleted = self.snake[-1]
		del self.snake[-1]
		self.warp()
		reward, terminal = self.check_collision()
		last_score = self.score
		last_time = self.time_alive
		if self.use_middle_point:
			observables = self.calculate_observables_half()
		else:
			observables = self.calculate_observables()
		return observables, reward, terminal, {'last_score': last_score, 'last_time': last_time}