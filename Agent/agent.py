import numpy as np
import random

import keras
from keras import Sequential

from keras import layers
from keras import optimizers
from keras import saving

from collections import deque 

class Agent:
	def __init__(self, state_size, actions = 3, memory=1000, is_eval=False, model_name="", lr=0.001,gamma = 0.95,epsilon = 1.0, epsilon_min= 0.01, epsilon_dec = 0.995):
		self.state_size = state_size # normalized previous days
		self.action_size = actions # sit, buy, sell
		self.memory = deque(maxlen=memory)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_dec
		print(f"lr: {lr}")
		self.model = saving.load_model("models/" + model_name) if is_eval else self._model(lr)

	def _model(self,learning_rate):
		model = Sequential()
		model.add(layers.Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(layers.Dense(units=32, activation="relu"))
		model.add(layers.Dense(units=8, activation="relu"))
		model.add(layers.Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=learning_rate))

		return model

	def act(self, state):
		if (not self.is_eval) and (np.random.rand() <= self.epsilon):
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 