"""
Inference of the case base with k=1
"""


from CBR import retrieval
from CBR import analogy
import numpy as np
import copy




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
        


def adaptation(A, B, C, harmony):
    D = analogy.solveAnalogy(A, B, C)[0][0][0]
    if harmony:
        if "a" in C or "o" in C or "u" in C:
            D = D.replace("ä", "a")
            D = D.replace("ö", "o")
            D = D.replace("y", "u")
        else:
            D = D.replace("a", "ä")
            D = D.replace("o", "ö")
            D = D.replace("u", "y")
    return D




def update_probas(x, y, probas_cb, probas_dist, X, Y, n_data, distances_def):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    
    ab = [[X[i], Y[i]] for i in range(n_data)]
    distances = [[d(case, x) for case in ab] for d in distances_def]
    order = [np.argsort(distances[d]) for d in range(3)]
    
    
    for i in range(n_data):
        # updating probas_cb[i]

        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            if analogy.solveAnalogy(X[j], Y[j], x)[0][0][0] != y: 
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
        if analogy.solveAnalogy(X[j], Y[j], x)[0][0][0] != y: 
            continue
        for d in range(3):
            likelihood_result[d] += proba_1nn_base(order[d], j, probas_cb)
    
    updated_probas_dist = probas_dist * likelihood_result 
    updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
    
    return updated_probas_cb, updated_probas_dist






def update_probas_2(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_data, distances_def):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    
    ab = [[X[i], Y[i]] for i in range(n_data)]
    distances = [[d(case, x) for case in ab] for d in distances_def]
    order = [np.argsort(distances[d]) for d in range(3)]
    
    
    for i in range(n_data):
        # updating probas_cb[i]

        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            
            # if harmony
            likelihood_adapt = 0
            if adaptation(X[j], Y[j], x, True) == y:
                 likelihood_adapt += proba_harmony
            if adaptation(X[j], Y[j], x, False) == y:
                 likelihood_adapt += (1 - proba_harmony)
            
            
            for d in range(3):    
                likelihood_result_1 += likelihood_adapt * proba_1nn(order[d], j, i, 1, probas_cb) * probas_dist[d]
                likelihood_result_0 += likelihood_adapt * proba_1nn(order[d], j, i, 0, probas_cb) * probas_dist[d]
        
        proba_1 = likelihood_result_1 * probas_cb[i]
        proba_0 = likelihood_result_0 * (1 - probas_cb[i])
        updated_probas_cb[i] = proba_1 / (proba_1 + proba_0)
        
    # Updating distance
    
    likelihood_result = np.array([0.0, 0.0, 0.0])
    for j in range(n_data):
        # if harmony
        likelihood_adapt = 0
        if adaptation(X[j], Y[j], x, True) == y:
            likelihood_adapt += proba_harmony
        if adaptation(X[j], Y[j], x, False) == y:
            likelihood_adapt += (1 - proba_harmony)
           
        for d in range(3):
            likelihood_result[d] += likelihood_adapt * proba_1nn_base(order[d], j, probas_cb)
     
    updated_probas_dist = probas_dist * likelihood_result 
    updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
            
    # Updating vowel harmony
    
    likelihood_result = np.array([.0,.0])
    for j in range(n_data):
        # if harmony
        works_harmony = adaptation(X[j], Y[j], x, True) == y
        works_non_harmony = adaptation(X[j], Y[j], x, False) == y
        
        if (works_harmony or works_non_harmony):
            for d in range(3):    
                likelihood_result[0] += works_non_harmony * proba_1nn_base(order[d], j, probas_cb) * probas_dist[d]
                likelihood_result[1] += works_harmony * proba_1nn_base(order[d], j, probas_cb) * probas_dist[d]
           
    
    
    
    updated_probas_harmony_1 = likelihood_result[1] * proba_harmony
    updated_probas_harmony_0 = likelihood_result[0] * (1 - proba_harmony)
    updated_probas_harmony_1 = updated_probas_harmony_1 / (updated_probas_harmony_1 + updated_probas_harmony_0)
    
    return updated_probas_cb, updated_probas_dist, updated_probas_harmony_1
        







###############################################################################


#CB_teach = [['koira', 'koiran'],
#            ['kissa', 'kissan'],
#            ['talo', 'talon'],
#            ['rakkaus', 'rakkauden'], 
#            ['ystäväys', 'ystäväyden'], 
#            ['tervellinen', 'tervellisen'],
#            ['keltainen', 'keltaisen'],
#            ['hyvärinen', 'hyvärisen'],
#            ['vieras', 'vieraan'],
#            ['sairas','sairaan']]



CB_teach = [['koira', 'koirassa'],
            ['mäyrä', 'mäyrässä'],
            ['talo', 'talossa'],
            ['rakkaus', 'rakkaudessa'], 
            ['ystäväys', 'ystäväydessä'], 
            ['tervellinen', 'tervellisessä'],
            ['keltainen', 'keltaisesssa'],
            ['hyvärinen', 'hyvärisessä'],
            ['vieras', 'vieraassa'],
            ['sairas','sairaassa']]


X = [case[0] for case in CB_teach]
Y = [case[1] for case in CB_teach]

n_words = len(CB_teach)

chosen_indices = [0,4,6]

harmony = True


CB_learn = []
for i in chosen_indices:
    CB_learn.append(CB_teach[i])
    
dist_learn = retrieval.dist5




distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]
n_dist = len(distances_def)

probas_cb = .5 * np.ones(n_words)
#probas_dist = np.ones(n_dist) / n_dist
probas_dist = np.array([.1,.1,.8])
proba_harmony = .5



for i in range(n_words):
    if i not in chosen_indices:
        x = X[i]
        source, _ = retrieval.retrieval(CB_learn, x, dist_learn)
        #y = analogy.solveAnalogy(source[0][0], source[0][1], x)[0][0][0]
        y = adaptation(source[0][0], source[0][1], x, harmony)
        
        #probas_cb, probas_dist = update_probas(x, y, probas_cb, probas_dist, X, Y, n_words, distances_def)
        probas_cb, probas_dist, proba_harmony = update_probas_2(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_words, distances_def)
        #print(probas_cb)

