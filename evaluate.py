import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import numpy as np

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

try:
	if len(sys.argv) != 3:
		print ("Usage: python evaluate.py [stock] [model]")
		exit()

	stock_name, model_name = sys.argv[1], sys.argv[2]
	model = load_model("models/" + model_name)
	window_size = model.layers[0].input.shape.as_list()[1]

	agent = Agent(window_size, True, model_name)
	data = getStockDataVec(stock_name)
	l = len(data) - 1
	batch_size = 32

	state = getState(data, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []

	# Setup our plot
	fig, ax = plt.subplots()
	timeseries_iter = 0
	#plt_data = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			#plt_data.append((timeseries_iter, data[t], 'Buy'))
			print ("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			#plt_data.append((timeseries_iter, data[t], 'Sell'))
			
			print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		#timeseries_iter += 1
		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print (stock_name + " Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")
		
		if len(agent.memory) > batch_size:
				agent.expReplay(batch_size) 

	#plt_data = np.array(plt_data)/
	#ax.plot(plt_data[:, 0], plt_data[:, 1])
	#Display our plots
	#plt.show()
except Exception as e:
	print("Error is: " + e)
finally:
	exit()
