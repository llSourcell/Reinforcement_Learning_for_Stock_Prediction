## Overview

This is the code for [this](https://www.youtube.com/watch?v=05NqKJ0v7EE) video on Youtube by Siraj Raval. The author of this code is [edwardhdlu](https://github.com/edwardhdlu/q-trader) . It's implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

## Results

Some examples of results on test sets:

![^GSPC 2015](https://github.com/edwardhdlu/q-trader/blob/master/images/^GSPC_2015.png)
S&P 500, 2015. Profit of $431.04.

![BABA_2015](https://github.com/edwardhdlu/q-trader/blob/master/images/BABA_2015.png)
Alibaba Group Holding Ltd, 2015. Loss of $351.59.

![AAPL 2016](https://github.com/edwardhdlu/q-trader/blob/master/images/AAPL_2016.png)
Apple, Inc, 2016. Profit of $162.73.

![GOOG_8_2017](https://github.com/edwardhdlu/q-trader/blob/master/images/GOOG_8_2017.png)
Google, Inc, August 2017. Profit of $19.37.

## Running the Code

To train the model, download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`
```
mkdir models
python train ^GSPC 10 1000
```

Then when training finishes (minimum 200 episodes for results):
```
python evaluate.py ^GSPC_2011 model_ep1000
```

## Some changes I had to modify from original
- Had to modify sigmoid slightly to prevent overflow occuring in Math.exp
- Added a boolean for first iteration which forces a buy so there is something in agent.inventory  (May look into finding the best statistical time to buy in a future imp based on the current price of entry...for now this has gotten things working when evaluating)

## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code
