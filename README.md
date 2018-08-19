# K-Means Clustering on Iris Dataset
K-means is an unsupervised learning algorithm, which tries to find clusters in an unlabeled dataset. <br />
The algorithm works as follows, assuming we have inputs x1,x2,x3,…,xn and value of K(which is 3 here) <br />

Step 1 - Pick K points as cluster centers called centroids. <br />
Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid. <br />
Step 3 - Find new cluster center by taking the average of the assigned points. <br />
Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change. <br />
