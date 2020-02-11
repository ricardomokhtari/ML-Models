#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:01:15 2020

@author: Ricardo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implement UCB algorithm
N = dataset.shape[0] # this is number of observations in dataset
d = dataset.shape[1] # this is number of independent variables
ads_selected = []    # keeps track of all the ads selected

numbers_of_selections = [0] * d # keeps track of the number of times that ad i was selected
sums_of_reward = [0] * d        # keeps track of the total reward at round n
total_reward = 0

"""

NOTE: before we go into the full algorithm, we first need to select all the 10
ads in the first 10 rounds so we have a datapoint on each one before proceeding
with the rest of the data

"""

for n in range(N):
    # initialise variables at each round
    ad = 0
    max_upper_bound = 0
    
    for i in range(d):
        if (numbers_of_selections[i] > 0):
            # compute the average reward of ad i up to round n
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            # compute the upper confidence interval based on formula
            delta_i =  math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # this part means that at first we make sure we
                                # select the first 10 ads in the first 10 rounds
                                # this is so we gather some data on all the ads
                                # to use the actual algorithm on the rest of the
                                # data
        
        # if the calcualted upper bound for that ad is greater than the current
        # max upper bound set the max upper bound to be the current upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i    # keep track of the ad this corresponds to this selection
            
    ads_selected.append(ad) # append selected ad
    numbers_of_selections[ad] += 1 # increment number of selections for this ad
    reward = dataset.values[n, ad] # check the corresponding ad in the original dataset
                                   # if the chosen ad was correct then reward is 1
                                   # if it was incorrect the reward is 0
    sums_of_reward[ad] += reward   # add reward to corresponding ad in array 
    total_reward += reward         # keep track of the total reward 

# visualise results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
        
    
