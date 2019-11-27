
# Actor-Critic Buy
import numpy as np
import random
from keras.models import load_model
from functions import *
from agent.agent import A2CAgent


data = getStockDataVec("taiW_test")
l = len(data) - 1
window_size = 10

actor_model = "models/model_actor-1514.hdf5"
critic_model = "models/model_critic-1514.hdf5"
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
		
		if action == 1: # buy
		#if action == 1 and len(agent.inventory) < 1: # buy Start & one position only
			agent.inventory.append(data[t])
			print(str(t)+" Buy: " + str(data[t]))


		elif action == 2 and len(agent.inventory) > 0: # sell
			tradeN += 1
			bought_price = agent.inventory.pop(0)
			profit = data[t] - bought_price
			total_profit += profit
			if profit > 0:
				    winN += 1
			print(str(t)+" Sell: " + str(data[t]) + " | Profit: " + str(data[t] - bought_price))
            
		else:
			print(str(t))
            
            
		done = True if t == l - 1 else False
		state = next_state

		if done:
			winNR = 100*winN/tradeN
			print ("--------------------------------")
			print (" Total Profit: " + formatPrice(total_profit))
			print ("Winning Rate: {:.2f} %".format(winNR))
			print ("Trade No: "+str(tradeN))
			print ("--------------------------------")
           
            
            
            
            
