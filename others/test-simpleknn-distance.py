import numpy as np
from scipy.spatial.distance import cdist
import copy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Secibd case: 
# - the CB is unknown
# - the distance is unknown


X, Y = make_blobs(n_samples=100, centers=12, n_features=2, cluster_std=.5,random_state=0)





n_data = X.shape[0]

probas_cb = .5 * np.ones(n_data)
probas_dist = np.ones(3) / 3.
        
# Distances:
# - d0: Euclidean distance on 0-axis
# - d1: Euclidean distance on 1-axis
# - d2: Euclidean distance on R^2



k = 1



def proba_1nn_base(order, j, probas_cb):
    proba = 1
    for l in order:
        if l == j: 
            return proba * probas_cb[j]
        else:
            proba *= (1 - probas_cb[l])
    

def proba_1nn(order, j, i, i_proba, probas_cb):
    probas_cb_new = copy.copy(probas_cb)
    probas_cb_new[i] = i_proba
    return proba_1nn_base(order, j, probas_cb_new)
        


def update_probas(x, y, probas_cb, probas_dist, X, Y, n_data):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    distances = [[d for d in np.abs(X[:,0] - x[:,0])], 
                  [d for d in np.abs(X[:,1] - x[:,1])], 
                  [xx[0] for xx in cdist(X, x)]] # x must be a 2d array
    order = [np.argsort(distances[d]) for d in range(3)]
    
    
    for i in range(n_data):
        # updating probas_cb[i]
        #print("---------------- ",  i)
        
        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            if Y[j] != y: 
                continue
            for d in range(3):    
                likelihood_result_1 += proba_1nn(order[d], j, i, 1, probas_cb) * probas_dist[d]
                likelihood_result_0 += proba_1nn(order[d], j, i, 0, probas_cb) * probas_dist[d]
        
        proba_1 = likelihood_result_1 * probas_cb[i]
        proba_0 = likelihood_result_0 * (1 - probas_cb[i])
        updated_probas_cb[i] = proba_1 / (proba_1 + proba_0)
        
    # Updating distance
    
    likelihood_result = np.array([0.0, 0.0, 0.0])
    for j in range(n_data):
        if Y[j] != y: 
            continue
        for d in range(3):
            #likelihood_result[d] += proba_1nn_base(order[d], j, probas_cb)
            likelihood_result[d] += proba_1nn_base(order[d], j, probas_cb)
    
    updated_probas_dist = probas_dist * likelihood_result 
    updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
    
    return updated_probas_cb, updated_probas_dist
        




# Learner's CB:
indices = list(range(n_data))
np.random.shuffle(indices)
n_cb = 10

chosen_indices = indices[:n_cb]

X_learner = X[chosen_indices,:]
Y_learner = Y[chosen_indices]



for i in range(n_data):
    if i not in chosen_indices:
        x = np.array([X[i,:],])
        distances = [xx[0] for xx in cdist(X_learner, x)]
        y = Y_learner[np.argmin(distances)]
        
        probas_cb, probas_dist = update_probas(x, y, probas_cb, probas_dist, X, Y, n_data)
        #print(probas_cb)




colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']
colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(20)]

for k, col in enumerate(colors):
    cluster_data = Y == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1],
                c=col, marker='.', s=10)

plt.scatter(X_learner[:,0], X_learner[:,1], marker='x', s=25)