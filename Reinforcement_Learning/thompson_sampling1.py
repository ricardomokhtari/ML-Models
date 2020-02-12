"""

This script is an implementation of the Thompson sampling reinforcement learning
technique.

The dataset used here is a variation of the multi armed bandit problem:
    - We have 10 different ads that we want to show to people
    - We don't know beforehand which ad is going to result in the most clicks
    - We implement Thompson sampling to find out
    - In the real world this is done as data is being gathered but in this situation
    we are not able to gather data on the fly so instead we have a pre made dataset
    (which you would not have in real life)
    - The dataset shows which ads were clicked by each user in each round (1 is clicked
    and 0 is not clicked)
    - We step through this data in our algorithm to simulate how this would be done in
    real life

"""

import pandas as pd
import matplotlib.pyplot as plt
import random

# import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implement Thompson Sampling
N = dataset.shape[0] # this is number of observations in dataset (no. of rounds)
d = dataset.shape[1] # this is number of independent variables (no. of iterations in each round)
ads_selected = []    # keeps track of all the ads selected

numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

sums_of_reward = [0] * d        # keeps track of the total reward at round n
total_reward = 0

for n in range(N):
    # initialise variables at each round
    ad = 0
    max_random = 0
    
    for i in range(d):
        # draw a random variable from the beta distribution
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        
        # if the value drawn from the beta distribution is greater than the current
        # max set the max radnom to be the current random
        if random_beta > max_random:
            max_random = random_beta
            ad = i    # keep track of the ad this corresponds to this selection
            
    ads_selected.append(ad) # append selected ad
        
    reward = dataset.values[n, ad] # check the corresponding ad in the original dataset
                                   # if the chosen ad was correct then reward is 1
                                   # if it was incorrect the reward is 0
    if reward == 0:
        numbers_of_rewards_0[ad] += 1
    else:
        numbers_of_rewards_1[ad] += 1
                                   
    total_reward += reward         # keep track of the total reward 

# visualise results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

