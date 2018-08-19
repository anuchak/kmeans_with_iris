from copy import deepcopy
import numpy as np
import pandas as pd
import random
data = pd.read_csv('iris.csv')
c1 = data['slength'].values
c2 = data['swidth'].values
c3 = data['plength'].values
c4 = data['pwidth'].values
X = np.array(list(zip(c1,c2,c3,c4)))
k=3
c1 = [X[0][0],X[1][0],X[2][0]]
c2 = [X[0][1],X[1][1],X[2][1]]
c3 = [X[0][2],X[1][2],X[2][2]]
c4 = [X[0][3],X[1][3],X[2][3]]
c = np.array(list(zip(c1,c2,c3,c4)), dtype=np.float32)
print(c)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

c_old = np.zeros(c.shape)
clusters = np.zeros(len(X))
error = dist(c,c_old,None)
while error!=0:
    for i in range(len(X)):
        distances = dist(X[i],c)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    c_old = deepcopy(c)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        c[i] = np.mean(points, axis=0)
    error = dist(c, c_old, None)
    
print(c)
print(clusters)
print(error)









