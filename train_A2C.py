
#Buy Start
from agent.agent import A2CAgent
from functions import *
import sys

stock_name = "taiW"
window_size = 10
episode_count = 2000

agent = A2CAgent(window_size, action_size=3)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
		print ("Episode " + str(e) + "/" + str(episode_count))
		state = getState(data, 0, window_size + 1)

		total_profit = 0
		tradeN = 0
		winN = 0
		agent.inventory = []

		for t in range(l):
			action = agent.act(state)

			# sit
			next_state = getState(data, t + 1, window_size + 1)
			reward = 0

			if action == 1: # buy
			#if action == 1 and len(agent.inventory) < 1: # buy Start & one position only
				agent.inventory.append(data[t])
				#print ("Buy: " + formatPrice(data[t]))

			elif action == 2 and len(agent.inventory) > 0: # sell
				tradeN += 1
				bought_price = agent.inventory.pop(0)
				reward = data[t] - bought_price
				profit = data[t] - bought_price        
				total_profit += profit
				if profit > 0:
				    winN += 1
				#print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
                
			done = True if t == l - 1 else False
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print ("--------------------------------")
				print ("Total Profit: " + formatPrice(total_profit))
				print ("--------------------------------")

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)


		if total_profit > 1500 and e > 200:
			agent.actor.save("models/model_actor-" + str(e)+".hdf5")
			agent.critic.save("models/model_critic-" + str(e)+".hdf5")
			f = open('models/rst.txt','a')
			f.write("No: "+str(e)+" Total Profit: " + formatPrice(total_profit)+'\n')
			try:
				winNR = 100*winN/tradeN
			except ZeroDivisionError:
			    f.write("No Trade!!!"+'\n') 
			else:             
			    f.write("Winning Rate: {:.2f} %".format(winNR)+'\n')
			    f.write("Trade No: "+str(tradeN)+'\n')
			f.write('====================================='+'\n')
			f.close()         





"""
#Sell Start
from agent.agent import A2CAgent
from functions import *
import sys

stock_name = "taiW"
window_size = 10
episode_count = 2000

agent = A2CAgent(window_size, action_size=3)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
		print ("Episode " + str(e) + "/" + str(episode_count))
		state = getState(data, 0, window_size + 1)

		total_profit = 0
		tradeN = 0
		winN = 0
		agent.inventory = []

		for t in range(l):
			action = agent.act(state)

			# sit
			next_state = getState(data, t + 1, window_size + 1)
			reward = 0			

			if action == 2: # sell
				tradeN += 1
				agent.inventory.append(data[t])
				#print ("Sell: " + formatPrice(data[t]))

			elif action == 1 and len(agent.inventory) > 0: # buy
				#tradeN += 1
				sold_price = agent.inventory.pop(0)
				reward = sold_price - data[t]
				profit = sold_price - data[t]        
				total_profit += profit
				if profit > 0:
				    winN += 1
				#print("Buy: " + formatPrice(data[t]) + " | Profit: " + formatPrice(sold_price - data[t]))
                
			done = True if t == l - 1 else False
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print ("--------------------------------")
				print ("Trade No: " + str(e))
				print ("Total Profit: " + formatPrice(total_profit))
				print ("--------------------------------")

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)


		if total_profit > 1500 and e > 200:
			agent.actor.save("models/model_actor-" + str(e)+".hdf5")
			agent.critic.save("models/model_critic-" + str(e)+".hdf5")
			f = open('models/rst.txt','a')
			f.write("No: "+str(e)+" Total Profit: " + formatPrice(total_profit)+'\n')
			try:
				winNR = 100*winN/tradeN
			except ZeroDivisionError:
			    f.write("No Trade!!!"+'\n') 
			else:             
			    f.write("Winning Rate: {:.2f} %".format(winNR)+'\n')
			    f.write("Trade No: "+str(tradeN)+'\n')
			f.write('====================================='+'\n')
			f.close()
"""