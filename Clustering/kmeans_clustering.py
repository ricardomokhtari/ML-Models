"""

Implemetation of K-Means clustering algorithm for 2 independent variables:
    
    - Annual income
    - Spending score
    
Optimal K is identified through minimising WCSS (elbow method)

K = 5

Initialised with K-Means++ algorithm

"""

# importing libraries
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# calculate optimal number of clusters
from sklearn.cluster import KMeans
wcss = []   # create empty list to hold WCSS values for each K-Means fit

# loop through 10 values of K to find the optimal value, calculate WCSS on each pass
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # kmeans.inertia_ is the WCSS calculation

# visualise the plot of WCSS - optimal K is found to be 5
plt.figure(1) 
plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()

# apply k means to mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10,
                    random_state=0)
y_kmeans = kmeans.fit_predict(X)

# visualise clustering
plt.figure(2)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', 
            label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', 
            label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', 
            label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'pink', 
            label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'black', 
            label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', 
            label = 'centroids')
plt.title('Clusters')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()