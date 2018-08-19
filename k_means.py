from copy import deepcopy
import numpy as np
import pandas as pd
import random

data = pd.read_csv('iris.csv') #reading from file
c1 = data['slength'].values    
c2 = data['swidth'].values
c3 = data['plength'].values
c4 = data['pwidth'].values
X = np.array(list(zip(c1,c2,c3,c4)))

k=3                            
#no. of clusters (3 in the sample data)

c1 = [X[0][0],X[1][0],X[2][0]] 
#first feature cluster centroids
c2 = [X[0][1],X[1][1],X[2][1]] 
#second feature cluster centroids
c3 = [X[0][2],X[1][2],X[2][2]] 
#third feature cluster centroids
c4 = [X[0][3],X[1][3],X[2][3]] 
#fourth feature cluster centroids

# A better way to initialize cluster centroids would be the kmeans++ algorithm

c = np.array(list(zip(c1,c2,c3,c4)), dtype=np.float32)
print(c)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
# returns the vector norm of cluster centroids and the corresponding features

# Stores the centroid values, as they are updated, initialiased to zeros
c_old = np.zeros(c.shape)
# Stores the centroid nearest to the point
clusters = np.zeros(len(X))
# Stores the error at this stage, iteration runs till error becomes zero
error = dist(c,c_old,None)
while error!=0:
    # Assigning each point to its nearest cluster
    for i in range(len(X)):
        distances = dist(X[i],c)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Old centroid values stored in c_old 
    c_old = deepcopy(c)
    # Finding the new mean of each cluster
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        c[i] = np.mean(points, axis=0)
    error = dist(c, c_old, None)
    
print(c)
print(clusters)
print(error)
