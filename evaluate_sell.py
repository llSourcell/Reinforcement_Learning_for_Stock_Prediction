# actor- critic
import keras
from keras.models import load_model

import numpy as np
import random
from keras.models import load_model
from functions import *
from agent.agent import A2CAgent

data = getStockDataVec("taiW_test")
l = len(data) - 1
window_size = 10

actor_model = "models/model_actor-206.hdf5"
critic_model = "models/model_critic-206.hdf5"
agent = A2CAgent(window_size, action_size=3,load_models = True, actor_model_file = actor_model, critic_model_file = critic_model)

total_profit = 0
agent.inventory = []
actionN = []
tradeN = 0
winN = 0

state = getState(data, 0, window_size + 1)
for t in range(l):
		action = agent.act(state)
		actionN.append(action)        

		# sit
		next_state = getState(data, t + 1, window_size + 1)


        # Must Change the Agent!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		if action == 2: # sell
			agent.inventory.append(data[t])
			#plt_data.append((timeseries_iter, data[t], 'Buy'))
			#print ("Buy: " + formatPrice(data[t]))
			print(str(t)+" Sell: " + str(data[t]))


		elif action == 1 and len(agent.inventory) > 0: # buy
			tradeN += 1
			sold_price = agent.inventory.pop(0)
			profit = sold_price-data[t]
			total_profit += profit				        
			if profit > 0:
				    winN += 1
            #plt_data.append((timeseries_iter, data[t], 'Sell'))
			#print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			#print(str(t)+" Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			print(str(t)+" Buy: " + str(data[t]) + " | Profit: " + str(sold_price-data[t]))
            
		else:
			print(str(t))
            
            
		#timeseries_iter += 1
		done = True if t == l - 1 else False
		#agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			winNR = 100*winN/tradeN
			print ("--------------------------------")
			print (" Total Profit: " + formatPrice(total_profit))
			print ("Winning Rate: {:.2f} %".format(winNR))
			print ("Trade No: "+str(tradeN))
			print ("--------------------------------")



"""
#
import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import numpy as np

stock_name = "taiW_test"
model_name = "model_ep797.hdf5"
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
actionN = []
tradeN = 0
winN = 0
		


for t in range(l):
		action = agent.act(state)
		actionN.append(action)        

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

        # Must Change the Agent!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		if action == 2: # sell
			agent.inventory.append(data[t])
			#plt_data.append((timeseries_iter, data[t], 'Buy'))
			#print ("Buy: " + formatPrice(data[t]))
			print(str(t)+" Sell: " + str(data[t]))


		elif action == 1 and len(agent.inventory) > 0: # buy
			tradeN += 1
			sold_price = agent.inventory.pop(0)
			profit = sold_price-data[t]
			total_profit += profit				        
			if profit > 0:
				    winN += 1
            #plt_data.append((timeseries_iter, data[t], 'Sell'))
			#print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			#print(str(t)+" Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			print(str(t)+" Buy: " + str(data[t]) + " | Profit: " + str(sold_price-data[t]))
            
		else:
			print(str(t))
            
            
		#timeseries_iter += 1
		done = True if t == l - 1 else False
		#agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			winNR = 100*winN/tradeN
			print ("--------------------------------")
			print (stock_name + " Total Profit: " + formatPrice(total_profit))
			print ("Winning Rate: {:.2f} %".format(winNR))
			print ("Trade No: "+str(tradeN))
			print ("--------------------------------")
"""		
