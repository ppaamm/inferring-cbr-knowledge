import numpy as np
from scipy.spatial.distance import cdist
import copy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# First case: Only the CB is unknown

X, Y = make_blobs(n_samples=200, centers=12, n_features=2, cluster_std=.5,random_state=0)





n_data = X.shape[0]

probas_cb = .5 * np.ones(n_data)

        





k = 1



def proba_1nn(order, j, i, i_proba, probas_cb):
    proba = 1
    probas_cb_new = copy.copy(probas_cb)
    probas_cb_new[i] = i_proba
    for l in order:
        if l == j: 
            return proba * probas_cb_new[j]
        else:
            proba *= (1 - probas_cb_new[l])
        


def update_probas(x, y, probas_cb, X, Y, n_data):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    distances = [xx[0] for xx in cdist(X, x)] # x must be a 2d array
    #print(distances)
    order = np.argsort(distances)
    
    
    for i in range(n_data):
        # updating probas_cb[i]
        #print("---------------- ",  i)
        
        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            if Y[j] != y: 
                #print('wrong class')
                continue
            likelihood_result_1 += proba_1nn(order, j, i, 1, probas_cb)
            likelihood_result_0 += proba_1nn(order, j, i, 0, probas_cb)
        
        proba_1 = likelihood_result_1 * probas_cb[i]
        proba_0 = likelihood_result_0 * (1 - probas_cb[i])
        updated_probas_cb[i] = proba_1 / (proba_1 + proba_0)
    return updated_probas_cb
        




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
        
        probas_cb = update_probas(x, y, probas_cb, X, Y, n_data)
        #print(probas_cb)




colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']
colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(20)]

for k, col in enumerate(colors):
    cluster_data = Y == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1],
                c=col, marker='.', s=10)

plt.scatter(X_learner[:,0], X_learner[:,1], marker='x', s=25)