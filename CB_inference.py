"""
Inference of the case base with k=1
"""


from CBR import analogy
import numpy as np
import copy



def proba_1nn_total(order, probas_cb):
    probas = np.zeros(len(order))
    proba = 1
    for l in order:
        probas[l] = proba * probas_cb[l]
        proba *= (1- probas_cb[l])
    return probas


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

def apply_harmony(C, D, harmony):
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


def init(A, B, C, distances_def):
    
    # compute solutions analogy
    a_solutions = [[[None for z in range(len(A))] for _ in range(len(C)) ] for h in (0,1)]
    for i in range(len(C)):
        for j in range(len(A)):
            a_solutions[0][i][j] = analogy.solveAnalogy(A[j], B[j], C[i])[0][0][0]
            a_solutions[1][i][j] = apply_harmony(C[i], a_solutions[0][i][j], True)

    # compute distances
    a_distances = np.zeros((len(distances_def), len(C), len(A)))
    for idx_d, d in enumerate(distances_def):
        for i in range(len(C)):
            a_distances[idx_d][i] = np.array([d([A[f], B[f]], C[i]) for f in range(len(A))])    
    
    a_orders = np.argsort(a_distances, axis=2)
    
    return a_solutions, a_distances, a_orders


def update_probas(x, y, probas_cb, probas_dist, X, Y, n_data, distances_def, dict_X, a_solutions, a_orders):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    
    # ab = [[X[i], Y[i]] for i in range(n_data)]
    # distances = [[d(case, x) for case in ab] for d in distances_def]
    # order = [np.argsort(distances[d]) for d in range(3)]
    
    idx_x = dict_X[x]
    order = [a_orders[d][idx_x] for d in range(len(distances_def))]
    
    for i in range(n_data):
        # updating probas_cb[i]

        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            # if analogy.solveAnalogy(X[j], Y[j], x)[0][0][0] != y: 
            if a_solutions[0][idx_x][j] != y: 
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
        if a_solutions[0][idx_x][j] != y: 
            continue
        for d in range(3):
            likelihood_result[d] += proba_1nn_base(order[d], j, probas_cb)
    
    updated_probas_dist = probas_dist * likelihood_result 
    updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
    
    return updated_probas_cb, updated_probas_dist






def update_probas_2(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_data, distances_def, dict_X, a_solutions, a_orders):
    """
    x: problem given to the user
    y: response of the user
    """
    
    updated_probas_cb = np.zeros(n_data)
    
    # Step 1: Re-order the CB by similarity (common step)
    
    # ab = [[X[i], Y[i]] for i in range(n_data)]
    # distances = [[d(case, x) for case in ab] for d in distances_def]
    # order = [np.argsort(distances[d]) for d in range(3)]
    
    idx_x = dict_X[x]
    order = [a_orders[d][idx_x] for d in range(len(distances_def))]
    
    for i in range(n_data):
        # updating probas_cb[i]

        # check lambda_i = 1 / 0 (case in CB)
        likelihood_result_1 = 0
        likelihood_result_0 = 0
        for j in range(n_data):
            
            # if harmony
            likelihood_adapt = 0
            # if adaptation(X[j], Y[j], x, True) == y:
            if a_solutions[1][idx_x][j] == y:
                 likelihood_adapt += proba_harmony
            # if adaptation(X[j], Y[j], x, False) == y:
            if a_solutions[0][idx_x][j] == y:
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
        # if adaptation(X[j], Y[j], x, True) == y:
        if a_solutions[1][idx_x][j] == y:
            likelihood_adapt += proba_harmony
        # if adaptation(X[j], Y[j], x, False) == y:
        if a_solutions[0][idx_x][j] == y:
            likelihood_adapt += (1 - proba_harmony)
           
        for d in range(3):
            likelihood_result[d] += likelihood_adapt * proba_1nn_base(order[d], j, probas_cb)
     
    updated_probas_dist = probas_dist * likelihood_result 
    updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
            
    # Updating vowel harmony
    
    likelihood_result = np.array([.0,.0])
    for j in range(n_data):
        # if harmony
        # works_harmony = adaptation(X[j], Y[j], x, True) == y
        # works_non_harmony = adaptation(X[j], Y[j], x, False) == y
        works_harmony = a_solutions[1][idx_x][j] == y
        works_non_harmony = a_solutions[0][idx_x][j] == y

        
        if (works_harmony or works_non_harmony):
            for d in range(3):    
                likelihood_result[0] += works_non_harmony * proba_1nn_base(order[d], j, probas_cb) * probas_dist[d]
                likelihood_result[1] += works_harmony * proba_1nn_base(order[d], j, probas_cb) * probas_dist[d]
           
    
    
    
    updated_probas_harmony_1 = likelihood_result[1] * proba_harmony
    updated_probas_harmony_0 = likelihood_result[0] * (1 - proba_harmony)
    updated_probas_harmony_1 = updated_probas_harmony_1 / (updated_probas_harmony_1 + updated_probas_harmony_0)
    
    return updated_probas_cb, updated_probas_dist, updated_probas_harmony_1
        
