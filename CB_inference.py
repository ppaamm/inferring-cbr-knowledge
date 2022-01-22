"""
Inference of the case base with k=1
"""


from CBR import analogy
import numpy as np
import copy
from typing import List




class PreComputation:
    
    def __init__(self, X, Y, X_teach, distances_def):
        a_s, a_d, a_o = self.run_precomputations(X, Y, X_teach, distances_def)
        self.a_solutions = a_s
        self.a_distances = a_d
        self.a_orders = a_o


    def run_precomputations(self, A, B, C, distances_def):    
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
        
        #a_orders = np.argsort(a_distances, axis=2)
        a_orders = [[order_duplicate(a_distances[d][x]) for x in  range(len(a_distances[d]))] for d in range(len(distances_def))]
        
        return a_solutions, a_distances, a_orders





class InferenceEngine:
    
    def __init__(self, CB: List[List[str]], 
                 prior_cb: np.ndarray, 
                 prior_dist: np.ndarray, 
                 prior_harmony: np.ndarray, 
                 precomputation: PreComputation):
        self.CB = CB
        self.probas_cb = prior_cb
        self.probas_dist = prior_dist
        self.proba_harmony = prior_harmony
        self.precomputation = precomputation
        
        self.n_data = self.probas_cb.shape[0]
        self.n_distances = self.probas_dist.shape[0]
        
        
        
    def _update_probas_cb(self, idx_x, y, order):
        updated_probas_cb = np.zeros(self.n_data)    
    
        
        
        for i in range(self.n_data):
            # updating probas_cb[i]
    
            # check lambda_i = 1 / 0 (case in CB)
            likelihood_result_1 = 0
            likelihood_result_0 = 0
            for j in range(self.n_data):
                
                # if harmony
                likelihood_adapt = 0

                if self.precomputation.a_solutions[1][idx_x][j] == y: 
                     likelihood_adapt += self.proba_harmony
                     
                if self.precomputation.a_solutions[0][idx_x][j] == y:
                     likelihood_adapt += (1 - self.proba_harmony)
    
                
                for d in range(3):                
                    likelihood_result_1 += likelihood_adapt * proba_1nn(order[d], j, i, 1, self.probas_cb) * self.probas_dist[d]
                    likelihood_result_0 += likelihood_adapt * proba_1nn(order[d], j, i, 0, self.probas_cb) * self.probas_dist[d]
    
            
            proba_1 = likelihood_result_1 * self.probas_cb[i]
            proba_0 = likelihood_result_0 * (1 - self.probas_cb[i])
            updated_probas_cb[i] = proba_1 / (proba_1 + proba_0)
        
        return updated_probas_cb
        
    
    
    def _update_probas_dist(self, idx_x, y, order):
        likelihood_result = np.zeros(self.n_distances)
        for j in range(self.n_data):
            # if harmony
            likelihood_adapt = 0
            
            if self.precomputation.a_solutions[1][idx_x][j] == y:
                likelihood_adapt += self.proba_harmony
                
            if self.precomputation.a_solutions[0][idx_x][j] == y:
                likelihood_adapt += (1 - self.proba_harmony)
               
            for d in range(3):
                likelihood_result[d] += likelihood_adapt * proba_1nn_base(order[d], j, self.probas_cb)
         
        updated_probas_dist = self.probas_dist * likelihood_result 
        updated_probas_dist = updated_probas_dist / np.sum(updated_probas_dist)
        
        return updated_probas_dist
    
    
    
    def _update_proba_harmony(self, idx_x, y, order):
        likelihood_result = np.array([.0,.0])
        for j in range(self.n_data):
            works_harmony = self.precomputation.a_solutions[1][idx_x][j] == y
            works_non_harmony = self.precomputation.a_solutions[0][idx_x][j] == y
    
            
            if (works_harmony or works_non_harmony):
                for d in range(3):    
                    likelihood_result[0] += works_non_harmony * proba_1nn_base(order[d], j, self.probas_cb) * self.probas_dist[d]
                    likelihood_result[1] += works_harmony * proba_1nn_base(order[d], j, self.probas_cb) * self.probas_dist[d]
                    
            updated_proba_harmony_1 = likelihood_result[1] * self.proba_harmony
            updated_proba_harmony_0 = likelihood_result[0] * (1 - self.proba_harmony)

            updated_proba_harmony = updated_proba_harmony_1 / (updated_proba_harmony_1 + updated_proba_harmony_0)
        
        return updated_proba_harmony
    
    
        
    
    def update_probas(self, idx_x: int, y: str):
        """
        Compute the posterior distribution given the user's given solution

        Parameters
        ----------
        idx_x : int
            Index of the source problem (in the global CB).
        y : str
            Given solution.

        Returns
        -------
        None.

        """
        order = [self.precomputation.a_orders[d][idx_x] for d in range(self.n_distances)]
        self.probas_cb = self._update_probas_cb(idx_x, y, order)
        self.probas_dist = self._update_probas_dist(idx_x, y, order)
        self.proba_harmony = self._update_proba_harmony(idx_x, y, order)
        
        
        

        
###############################################################################   
     
        
        
        




def proba_1nn_total(order, probas_cb):
    length = sum(len(L) for L in order)
    probas = np.zeros(length)
    proba = 1
    #TODO: 
    for L in order:
        for k in L: probas[k] = proba * probas_cb[k] / len(L)
        for k in L: proba *= (1- probas_cb[k])
    return probas



def proba_1nn_base(order, j, probas_cb, n_mc=100):
    proba = 1
    for L in order:
        if j in L: 
            #TODO
            return proba * probas_cb[j] / len(L)
        else:
            for k in L: proba *= (1 - probas_cb[k]) 
   

def proba_1nn(order, j, i, i_proba, probas_cb):
    probas_cb_new = copy.copy(probas_cb)
    probas_cb_new[i] = i_proba
    return proba_1nn_base(order, j, probas_cb_new)

             
                

def adaptation(A, B, C, harmony):
    """
    Deprecated
    """
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



def order_duplicate(L):
    """
    Returns the index of the elements of the list sorted from the smallest to
    the largest. Elements with a same value are grouped together in a list. 
    """
    order = np.argsort(L)
    prev = -1
    result = []
    for i in order:
        if L[i] == prev:
            result[-1].append(i)
        else:
            prev = L[i]
            result.append([i])
    return result







def update_probas_states(x, y, probas_state, X, Y, n_data, states, dict_X, a_solutions, a_orders):
    """
    x: problem given to the user
    y: response of the user
    """
    
    n_states = probas_state.shape[0]
    updated_probas = np.zeros(n_states)
    
    y_i = []
    
    idx_x = dict_X[x]
    
    for i in range(n_states):
        # Is CBR(x,s[i]) = y?
        distance = states[i]['distance']
        harmony = states[i]['harmony']
        cases = states[i]['cases']
        
        for J in a_orders[distance][idx_x]:
            #print(J)
            for j in J:
                #print(j)
                if j in cases:
                    #NN = a_orders[distance][idx_x][j]
                    #print(NN)
                    y_i.append(a_solutions[harmony][idx_x][j])
                
                if len(y_i) > 0: break
            
        if y in y_i:
            updated_probas[i] = y_i.count(y) / len(y_i)
    
    updated_probas = updated_probas * probas_state
    
    return updated_probas / np.sum(updated_probas)

     



def probabilistic_state_transition(x, probas_cb, dict_X, p):
    """
    x: New problem
    probas_cb: List of probabilities for each case to be in the CB
    dict_X: Indices of cases in CB
    p: Probability of retention
    """
    idx_x = dict_X[x]
    probas_cb[idx_x] = probas_cb[idx_x] + (1 - probas_cb[idx_x]) * p
    return probas_cb








def evaluate(X_test, Y_test, a_solutions, a_distances, a_orders, CB_user, distances_def, probas_cb, probas_dist, proba_harmony):
    score = 0
    
    for tgt in range(len(X_test)):
        list_y = Y_test[tgt]
        
        order = [a_orders[d][tgt] for d in range(len(distances_def))]
        probas = [proba_1nn_total(o, probas_cb) for o in order]
        
        for y, occ_y in list_y:
            p = 0
            for h in range(2):
                sol = a_solutions[h][tgt]
                if y in sol:
                    indices = [idx for idx, f in enumerate(sol) if f == y] # indexes of NN yielding solution y
                    for v in indices:
                        for d in range(len(distances_def)):
                            if (h==0):
                                p += (1 - proba_harmony) * probas_dist[d] * probas[d][v]
                            else:
                                p += proba_harmony * probas_dist[d] * probas[d][v]
            score += p * occ_y / len(list_y)
    return score / len(X_test)




def compare_probas(probas_cb, probas_cb_user):
    return np.sum(np.abs(probas_cb_user - probas_cb)) / probas_cb.shape[0]