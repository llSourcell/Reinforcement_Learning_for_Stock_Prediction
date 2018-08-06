import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from keras.callbacks import TensorBoard, EarlyStopping

import numpy as np
import random
import os
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.firstIter = True

		#Make sure our directory is available
		self.setupTensorboardDir()
		self.callbacks = [LRTensorBoard(log_dir=self.log_dir)]

		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	#Action
	def act(self, state):
		rand_val = np.random.rand()
		if not self.is_eval and rand_val <= self.epsilon:
			return random.randrange(self.action_size)

		if(self.firstIter):
			self.firstIter = False
			return 1
		options = self.model.predict(state)
		#print("Using prediction")
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
			self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=self.callbacks)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 

	def setupTensorboardDir(self):
		current_dir = os.path.dirname(os.path.realpath(__file__))
		self.log_dir = os.path.join(current_dir, '../tensorboard')
		print('Working directory: %s' % current_dir)
		if not os.path.exists(self.log_dir) or not os.path.isdir(self.log_dir):
			os.mkdir(self.log_dir)

class LRTensorBoard(TensorBoard):
	def __init__(self, *args, **kwargs):
		#self.scalar = kwargs.pop('scalar', True)
		super(LRTensorBoard, self).__init__(*args, **kwargs)

		global tf
		import tensorflow as tf

	def on_epoch_end(self, epoch, logs=None):
		logs.update({'lr': K.eval(self.model.optimizer.lr)})
		super(LRTensorBoard, self).on_epoch_end(epoch, logs)
