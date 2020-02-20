"""

This script is an implementation of the Apriori association rule learning algorithm.
It optimises the sales in a grocery store.

Dataset is in the form of
    - Rows: transactions
    - Columns: items bought in that transation

"""

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []

# need to parse dataset into list of lists of purchased items
# because this is the format the apriori algorithm accepts
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])
    
# train apriori on the dataset
from apyori import apriori

rules = apriori(transactions, 
                min_support = 0.003,    # products that are purchased 3 times per day * 7 / 7500 =~ 0.003
                min_confidence = 0.2,   # if confidence is too high (0.8) we will look at items that are
                                        # bought very often together but not necessarily related logically
                min_lift = 3, 
                min_length = 2)
        
# visualise the results
results = list(rules)

# need to parse results list to extract the relevant information from RelationRecord objects
results_list = []

for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + 
                        str(results[i][1]) + '\nInfo:\t' + str(results[i][2]))