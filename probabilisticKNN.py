import numpy as np

def probabilisticKNN(k, probas, distances, n_simu=100):
    if k == 1: return probabilistic1NN(probas, distances)
    return probabilistickNN_MC(k, probas, distances, n_simu)



# Exact result for k=1
# Returns the probability that 1NN returns each point of the CB
def probabilistic1NN(probas, distances):
    N = len(probas)
    order = np.argsort(distances)
    result_proba = [0 for _ in range(N)]
    
    failure_proba = 1
    
    for i in range(N):
        index = order[i]
        result_proba[index] =  failure_proba * probas[index]
        failure_proba *= (1-probas[index])
    return result_proba



# Monte-Carlo approximation
# Returns the probability that kNN returns each point of the CB
def probabilistickNN_MC(k, probas, distances, n_simu):
    N = len(probas)
    order = np.argsort(distances)
    
    result_proba = [0 for _ in range(N)]
    for _ in range(n_simu):
        cb = drawCB(probas)
        nn = []
        i = 0
        while len(nn) < k and i < N:
            if cb[order[i]] == 1:   ### Doesn't work => need to check that the index of i in order is... 
                nn.append(order[i])
                result_proba[order[i]] += 1./n_simu
            i += 1

    return result_proba


# Randomly draws a CB using the specified probabilities
def drawCB(probas):
    if all(p == 0 for p in probas): return probas
    while(True):
        result = np.random.binomial(n=1,p=probas)
        if max(result) > 0: return result




##### Example

from CBR import retrieval

CB = [['koira', 'koiran'], ['alue', 'alueen'], ['ruis', 'rukiin'], ['rakkaus', 'rakkauden'], ['lato', 'ladon']]


probas = [.2, .5, .3, .7, .9]
new_problem = "kissa"
distances = [retrieval.dist5(case, new_problem) for case in CB]

print(probabilistickNN_MC(1, probas, distances, 5000))
print(probabilistickNN_MC(2, probas, distances, 5000))
print(probabilistickNN_MC(3, probas, distances, 5000))
