"""

Implemetation of Hierarchical clustering algorithm for 2 independent variables:
    
    - Annual income
    - Spending score
    
Optimal cluster number is identified through examining dendrogram

"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
plt.figure(1)
dendrogram = sch.dendrogram(sch.linkage(X,method='ward')) # create dendrogram plot

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.figure(2)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', 
            label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', 
            label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', 
            label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'pink', 
            label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'black', 
            label = 'Cluster 5')
plt.title('Clusters')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()