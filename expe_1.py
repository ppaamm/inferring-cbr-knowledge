from CBR import retrieval
from CB_inference import adaptation, update_probas_full, proba_1nn_total, init
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time

###############################################################################
# The user has a fixed CB. 
# The agent infers the content of the CB + distance + vowel harmony
# Retrieval is using 1-NN
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
    
    distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]
    n_dist = len(distances_def)
    
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
    dict_X = {X[i]:i for i in range(len(X))}
    n_words = len(CB_teach)
    
    # Configuration of the learner

    chosen_indices = [0,4,6]
    harmony = True
    
    CB_learn = [CB_teach[i] for i in chosen_indices]
    dist_learn = distances_def[2]
    
    a_solutions, a_distances, a_orders = init(X, Y, X, distances_def)
    
    # Initalize priors
    probas_cb = .5 * np.ones(n_words)
    probas_dist = np.ones(n_dist) / n_dist
    #probas_dist = np.array([.1,.1,.8])
    proba_harmony = .5
    
    for i in range(n_words):
        print("word", i)
        if i not in chosen_indices:
            x = X[i]
            source, _ = retrieval.retrieval(CB_learn, x, dist_learn)
            # y = analogy.solveAnalogy(source[0][0], source[0][1], x)[0][0][0]
            # y = adaptation(source[0][0], source[0][1], x, harmony)
            idx_source = dict_X[source[0][0]]
            y = a_solutions[harmony][i][idx_source]
            #probas_cb, probas_dist = update_probas(x, y, probas_cb, probas_dist, X, Y, n_words, distances_def, dict_X, a_solutions, a_orders)
            probas_cb, probas_dist, proba_harmony = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_words, distances_def, dict_X, a_solutions, a_orders)

    return probas_cb, probas_dist, proba_harmony
    


###############################################################################


def evaluate(X_s, Y_s, X_t, Y_t, a_solutions, a_distances, a_orders, CB_user, distances_def, probas_cb, probas_dist, proba_harmony):
    score = 0
    
    for tgt in range(len(X_s)):
        y = Y_t[tgt]
        
        order = [a_orders[d][tgt] for d in range(len(distances_def))]
        probas = [proba_1nn_total(o, probas_cb) for o in order]
        
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
        score += p
    return score / len(X_s)


def evaluate2(CB_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony):
    score = 0
    for case in CB_test:
        x = case[0]
        y = case[1]
        
        # Retrieval 
        distances = [[d(learnt_case, x) for learnt_case in CB_user] for d in distances_def]
        order = [np.argsort(distances[d]) for d in range(3)]
        probas = [proba_1nn_total(o, probas_cb) for o in order]

        results = {}
        
        for i in range(len(CB_user)):
            print(i)
            for d in range(len(distances)):
                result_h_1 = adaptation(CB_user[i][0], CB_user[i][1], x, True)
                p = proba_harmony * probas_dist[d] * probas[d][i]
                if result_h_1 in results:
                    results[result_h_1] += p
                else: results[result_h_1] = p
                
                result_h_0 = adaptation(CB_user[i][0], CB_user[i][1], x, False)
                p = (1 - proba_harmony) * probas_dist[d] * probas[d][i]
                if result_h_0 in results:
                    results[result_h_0] += p
                else: results[result_h_0] = p
        if y in results: score += results[y]
    return score / len(CB_test)






###############################################################################
#### Pipeline

# Data generation:

start_time = time.time()

df = pd.read_csv ("./data/FI/Genitive/gen.txt")
df = df[df.genitive != '—']
df = df[df.genitive != '–']


type2 = df[df.type == 2]
type11 = df[df.type == 11]
type38 = df[df.type == 38]
type41 = df[df.type == 41]


# Parameters

distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]


# User definition:

n_user = 5

# CB contains all the possible cases that the user could know
CB = pd.concat([type2.sample(n=n_user),
                type11.sample(n=n_user),
                type38.sample(n=n_user),
                type41.sample(n=n_user)])

CB = CB.filter(items = ['nominative','genitive']).values.tolist()  
X = [z[0] for z in CB]
Y = [z[1] for z in CB]
n_words = len(CB)

# TODO: Choice of indices
known_indices = [0,5,10]

# CB_user contains the cases that the user actually knows
CB_user = [CB[i] for i in known_indices]

X_user = [z[0] for z in CB_user]
Y_user = [z[1] for z in CB_user]

distance_user = distances_def[2]
harmony_user = True


# Initialization of the test:

n_test = 2

CB_test = pd.concat([type2.sample(n=n_test),
                       type11.sample(n=n_test),
                       type38.sample(n=n_test),
                       type41.sample(n=n_test)])

CB_test = CB_test.filter(items = ['nominative','genitive']).values.tolist()  


X_test = [z[0] for z in CB_test]
Y_test = [z[1] for z in CB_test]

a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)






# Teaching




n_teach = 5
CB_teach = pd.concat([type2.sample(n=n_teach),
                      type11.sample(n=n_teach),
                      type38.sample(n=n_teach),
                      type41.sample(n=n_teach)])

CB_teach = CB_teach.filter(items = ['nominative','genitive']).values.tolist()
n_words_teach = len(CB_teach)
X_teach = [z[0] for z in CB_teach]
dict_X = {X_teach[i]:i for i in range(len(X_teach))}

a_solutions, a_distances, a_orders = init(X, Y, X_teach, distances_def)



runs = []
steps = []
p_harmony = []
p_d0 = []
p_d1 = []
p_d2 = []


n_runs = 20

for r in range(n_runs):
    print(r)
    
    # Priors

    probas_cb = .5 * np.ones(len(CB))
    probas_dist = np.ones(len(distances_def)) / len(distances_def)
    proba_harmony = .5
    
    
    randomized = random.sample(list(range(n_words_teach)), n_words_teach)
    runs.append(r)
    steps.append(0)
    p_harmony.append(proba_harmony)
    p_d0.append(probas_dist[0])
    p_d1.append(probas_dist[1])
    p_d2.append(probas_dist[2])
    
    for i in range(n_words_teach):
        #print("word", i)

        
        x = CB_teach[randomized[i]][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        probas_cb, probas_dist, proba_harmony = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_words, distances_def, dict_X, a_solutions, a_orders)
        
        runs.append(r)
        steps.append(i+1)        
        p_harmony.append(proba_harmony)
        p_d0.append(probas_dist[0])
        p_d1.append(probas_dist[1])
        p_d2.append(probas_dist[2])


# Plot


print("--- %s seconds ---" % (time.time() - start_time))
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'p_d0': p_d0, 'p_d1': p_d1, 'p_d2': p_d2})

sns.lineplot(x='step', y='value', hue='variable', 
             data=pd.melt(df, id_vars = ['step', 'run']))



# ------- Only for testing
#distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]

# -------


#print("start")
#print(evaluate(CB_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))
#print("--- %s seconds ---" % (time.time() - start_time))
