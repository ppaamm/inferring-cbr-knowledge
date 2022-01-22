from CB_inference import proba_1nn_total, PreComputation, InferenceEngine
import numpy as np


def compare_probas(probas_cb, probas_cb_user):
    return np.sum(np.abs(probas_cb_user - probas_cb)) / probas_cb.shape[0]

class Evaluation:
    def __init__(self, X_test, Y_test, precomputation: PreComputation):
        self.n_x = len(X_test)
        self.Y_test = Y_test
        self.precomputation = precomputation

    
    def evaluate(self, inference: InferenceEngine):
        score = 0
        
        for tgt in range(self.n_x):
            list_y = self.Y_test[tgt]
            
            order = [self.precomputation.a_orders[d][tgt] for d in range(inference.n_distances)]
            probas = [proba_1nn_total(o, inference.probas_cb) for o in order]
            
            for y, occ_y in list_y:
                p = 0
                for h in range(2):
                    sol = self.precomputation.a_solutions[h][tgt]
                    if y in sol:
                        indices = [idx for idx, f in enumerate(sol) if f == y] # indexes of NN yielding solution y
                        for v in indices:
                            for d in range(inference.n_distances):
                                if (h==0):
                                    p += (1 - inference.proba_harmony) * inference.probas_dist[d] * probas[d][v]
                                else:
                                    p += inference.proba_harmony * inference.probas_dist[d] * probas[d][v]
                score += p * occ_y / len(list_y)
        return score / self.n_x
    
    
    
    
