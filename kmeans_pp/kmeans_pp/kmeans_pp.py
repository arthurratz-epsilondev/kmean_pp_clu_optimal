#-----------------------------------------------------------------------------------
#   K-Means++ Optimal Data Clustering Algorithm v.0.0.1
#
#        C,S = kmeans_pp(N,k)
#
#        N - # of observations, k - # of clusters
#
#        The worst-case complexity of the K-Means++ procedure:
#
#                   
#
#                   An Example: M = 10^3, N = 10^2, p = 2.19 x 1e+6
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#-----------------------------------------------------------------------------------

import math
import time
import random
import numpy as np
import pandas as pd
import numpy.linalg as lin
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

def gen_dataset(n,d,k):
    # Generate an arbitrary dataset X as a matrix (n x d) of random elements [0;1)
    return np.random.random_sample((n,d))

def gen_dataset_synth(n,d,k):
    # Generate a synthetic dataset X as a distribution of n-points
    # having d-features, arranged into k-clusters with random standard deviation [0.75;1.0)
    return make_blobs(n_samples=n, n_features=d, \
        centers=k, cluster_std=np.random.uniform(0.75, 1.0), random_state=1)[0]

def exists(E,i):
    # return 'True' if the point i exists 
    # in the array E, and 'False' unless otherwise
    return 0 < np.shape(np.array( \
        [ e for e in np.array(E) if (e == i).all() ]))[0]

def eucld(i1,i2):
    # Compute the squared Euclidean distance d=|i1-i2|^2 as the sum of squared 
    # distances between points i1 and i2, at each dimension
    return np.sum(np.array([ \
        math.pow(i1 - i2, 2.0) \
          for i1, i2 in zip(i1, i2) ]))

def initialize(X):
    # Get the random centroid c0
    c0 = np.random.randint(0, np.shape(X)[0] - 1) + 1

    # Compute the distance from centroid c0 to each point in X
    c0_d = np.array([ eucld(X[c0], x) for x in X ])
    
    # Get the centroid c1's as one of the points in X,
    # having the maximum distance to the centroid c0
    c1 = np.where(c0_d >= np.max(c0_d))[0][0]
    
    return np.array([c0,c1]) # Return the indexes of c0 and c1

def compute(X,k):
    X = np.array(X)    # X - an input dataset of n-observations
    C = initialize(X)   # C - an initial set of centroids
    
    # Perform the dataset clustering iteratively, 
    # until the resultant set of k-clusters has been computed
    
    while True:
        S = np.empty(0)  # S - a set of newly built clusters
        
        # For each observation x[t] in X, do the following:
        for t in range(np.shape(X)[0]):
            # Check if the observation x[t] has already been
            # selected as one of the new centroids
            if exists(C, t) == False:
                # If not, compute the distance from 
                # the observation x[t] to each of the existing centroids in C
                cn_ds = np.array([ eucld(X[t], X[c]) for c in C ])
                # Get the centroid c[r] for which the distance to x[t] is the smallest
                cn_min_di = np.where(cn_ds == np.min(cn_ds))[0][0]

                # Assign the observation x[t] to the new cluster s[r], appending
                # the observation x[t]'s and centroid c[r]'s indexes to the set S
                S = np.append(S, { 'c': cn_min_di, 'i': t, 'd': cn_ds[cn_min_di] })

        # Terminate the clustering process, if the number of centroids 
        # in C is equal to the total number of clusters k, initially specified.
        
        # Otherwise, compute the next centroid c[r] in C
                
        if np.shape(C)[0] >= k: break
        
        # Get the distances |x-c[r]| from each observation 
        # to the centroid c[r], accross all existing clusters in S
        cn_ds = np.array([s['d'] for s in S ])
        
        # Compute the index of an observation, for which 
        # the distance to one of the centroids in C is the largest
        cn_max_ci = np.where(cn_ds == np.max(cn_ds))[0][0]

        # Append the index of a new centroid c[r] to the set C
        C = np.append(C, S[cn_max_ci]['i'])

    return C,S

def intra_cluster_d(X,S,r):
    di = 0; count = 0
    # For each cluster in S compute within-cluster average distance
    # as the sum of the observation distances to the centroid C[r]
    # divided by their quantity
    for i in range(np.shape(S)[0]):
        # Compute the distance between C[i] and C[j], 
        # accumulating it to the variable di
        if S[i]['c'] == r: 
            di = di + math.sqrt(S[i]['d']); 
            count = count + 1
    
    return float(di) / count # Return an average of the distances

def inter_cluster_d(X,C):
    di = 0; count = np.shape(C)[0]
    # Get the number of inter-cluster distances 
    # between pairs of centroids c[i] and c[j]
    count = math.pow(count, 2.0) - count
    # For each pair of centroids c[i] and c[j]
    for i in range(np.shape(C)[0]):
        for j in range(np.shape(C)[0]):
            # Accumulate their distances to the variable di
            di = di + math.sqrt(eucld(X[C[i]], X[C[j]]))
            
    return float(di) / count  # Return an average of the distances

n_observ       = 100                     # The number of observations
n_dims         = 2                       # The number of features (i.e., dimensions)
k_clusters     = 3                       # The total amount of clusters

ck_colors      = ['red','green','blue']  # Colors of points within each cluster

app_banner     = "K-Means++ Optimal Clustering CPOL License (C) 2021 by Arthur V. Ratz\n"
ds_stats       = "[ # Of Observations |X|: %d # Of Clusters k: %d ]\n"
obs_data       = "[ x = %f y = %f ]"
    
def plot(X,plt,clr):
    for i in range(np.shape(X)[0]):
        plt.plot(X[i][0], X[i][1], color=clr, linestyle='dashed', linewidth = 3,
            marker='o', markerfacecolor=clr, markersize=12)
        
    return plt

def output_dataset(X,name,type_s):
    
    print("[ Dataset %s: Type: %s ]\n" % (name, type_s))
    
    for x,i in zip(X,range(np.shape(X)[0])):
        print("%d: " % i, " ", obs_data % (x[0], x[1]))
    
    plot(X, plt, 'navy'); plt.show()
    
def output_clusters(X,C,S,plt):

    print("\n[ Centroids: ]\n")
    for c,i in zip(C,range(np.shape(C)[0])):
        x = X[c][0]; y = X[c][1]
        print("%d: " % i, " ", obs_data % (x, y))
    
    plt = plot(X, plt, 'navy')
    plot(np.array([X[c] for c in C]), plt, 'black')
    
    plt.show()
    
    for ci in range(k_clusters):
        di_avg = intra_cluster_d(X,S,ci)
        cx = X[C[ci]][0]; cy = X[C[ci]][1]
        print("\n[ Cluster: %d Centroid: %s (Average |di| = %f) ]\n" \
              % (ci, obs_data % (cx, cy), di_avg))
        for si in range(np.shape(S)[0]):
            if S[si]['c'] == ci:
                obs_x = X[S[si]['i']]
                print("%s |dc| = %f" % (obs_data % \
                    (obs_x[0], obs_x[1]), S[si]['d']))

    ds_avg = inter_cluster_d(X,C)
    
    for ci in range(k_clusters):
        Xs = np.array([ X[s['i']] for s in S if s['c'] == ci ])
        plt = plot(Xs, plt, ck_colors[ci])
        
    plt = plot(np.array([X[c] for c in C]), plt, 'black')
    
    plt.show()
    
    print("[ Average |ds| = %f ]\n" % ds_avg)
    
def main():
    
    print(app_banner)
    
    np.random.seed(int(time.time()))
    
    print(ds_stats % (n_observ, k_clusters))
    
    X1 = gen_dataset(n_observ,n_dims,k_clusters)
    X2 = gen_dataset_synth(n_observ,n_dims,k_clusters)
    
    output_dataset(X1, "X1", "Regular")

    C1,S1 = compute(X1, k_clusters)
    
    output_clusters(X1,C1,S1,plt)
    
    print("=======================================================================================================\n")
    
    output_dataset(X2, "X2", "Synthetic")

    C2,S2 = compute(X2, k_clusters)
    
    output_clusters(X2,C2,S2,plt)

if __name__=="__main__":
    main()