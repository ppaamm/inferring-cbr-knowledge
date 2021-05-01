from CBR import retrieval
from CBR import analogy
from CB_inference import adaptation, update_probas_2, proba_1nn_total
import numpy as np
import pandas as pd

###############################################################################


def toy():

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
    





df = pd.read_csv (".\data\FI\Genitive\gen.txt")
df = df[df.genitive != '—']
df = df[df.genitive != '–']


type2 = df[df.type == 2]
type11 = df[df.type == 11]
type38 = df[df.type == 38]
type41 = df[df.type == 41]


n_user = 2

CB_user = pd.concat([type2.sample(n=n_user),
                     type11.sample(n=n_user),
                     type38.sample(n=n_user),
                     type41.sample(n=n_user)])

CB_user = CB_user.filter(items = ['nominative','genitive'])        

n_teach = 5        
CB_teach = pd.concat([type2.sample(n=n_teach),
                      type11.sample(n=n_teach),
                      type38.sample(n=n_teach),
                      type41.sample(n=n_teach)])

CB_teach = CB_teach.filter(items = ['nominative','genitive'])      
        
n_test = 10

CB_test = pd.concat([type2.sample(n=n_test),
                     type11.sample(n=n_test),
                     type38.sample(n=n_test),
                     type41.sample(n=n_test)])

CB_test = CB_test.filter(items = ['nominative','genitive'])




def evaluate(CB_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony):
    score = 0
    for case in CB_test:
        x = case[0]
        y = case[1]
        
        # Retrieval 
        distances = [[d(learnt_case, x) for learnt_case in CB_user] for d in distances_def]
        order = [np.argsort(distances[d]) for d in range(3)]
        probas = [proba_1nn_total(o, probas_cb) for o in order]
        
        
    return 0