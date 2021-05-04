from CBR import retrieval
from CB_inference import adaptation, update_probas_full, init, evaluate
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt
import datasaver
import datetime



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
#### Pipeline

# Data generation:

start_time = time.time()

print("Loading data")

df = pd.read_csv ("./data/FI/Inessive/ine.txt")
df = df[df.inessive != '—']
df = df[df.inessive != '–']


type2 = df[df.type == 2]
type11 = df[df.type == 11]
type38 = df[df.type == 38]
type41 = df[df.type == 41]


# Parameters

distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]


# User definition:



distance_user = distances_def[2]
harmony_user = True


# Initialization of the test:

print("Initialization of the test")

n_test = 1

CB_test = pd.concat([type2.sample(n=n_test),
                       type11.sample(n=n_test),
                       type38.sample(n=n_test),
                       type41.sample(n=n_test)])

CB_test = CB_test.filter(items = ['nominative','inessive']).values.tolist()  


X_test = [z[0] for z in CB_test]
Y_test = [z[1] for z in CB_test]








print("Starting the teaching")


runs = []
steps = []
p_harmony = []
p_d0 = []
p_d1 = []
p_d2 = []
scores = []

n_runs = 10

for r in range(n_runs):
    print(r)
    
    
    ###########################################################################
    print("Creation of the user")

    n_user = 1
    
    # CB contains all the possible cases that the user could know
    CB = pd.concat([type2.sample(n=n_user),
                    type11.sample(n=n_user),
                    type38.sample(n=n_user),
                    type41.sample(n=n_user)])
    
    CB = CB.filter(items = ['nominative','inessive']).values.tolist()  
    X = [z[0] for z in CB]
    Y = [z[1] for z in CB]
    n_words = len(CB)
    
    # TODO: Choice of indices
    known_indices = [0, 1, 2]
    
    # CB_user contains the cases that the user actually knows
    CB_user = [CB[i] for i in known_indices]
    
    X_user = [z[0] for z in CB_user]
    Y_user = [z[1] for z in CB_user]
    
    
    ###########################################################################
    # Testing the user
    
    print("Evaluation of the user")
    
    a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)

    # Evaluating the user on the test base 
    
    Y_test_user = []
    for x in X_test:
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        Y_test_user.append(adaptation(source[0][0], source[0][1], x, harmony_user))
        
        
    ###########################################################################
    # Teacher's CB
    
    print("Initialisation of the teaching corpus")

    n_teach = 2
    CB_teach = pd.concat([type2.sample(n=n_teach),
                          type11.sample(n=n_teach),
                          type38.sample(n=n_teach),
                          type41.sample(n=n_teach)])
    
    CB_teach = CB_teach.filter(items = ['nominative','genitive']).values.tolist()
    n_words_teach = len(CB_teach)
    X_teach = [z[0] for z in CB_teach]
    dict_X = {X_teach[i]:i for i in range(len(X_teach))}
    
    a_solutions, a_distances, a_orders = init(X, Y, X_teach, distances_def)
        
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
    scores.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))
    
    
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
        scores.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))


# Plot


print("--- %s seconds ---" % (time.time() - start_time))


dt_string = 'expe1-' +  datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


# Figure 1

plt.figure()
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2})

sns.lineplot(x='step', y='probability', hue='distance', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='probability', var_name='distance'))

plt.savefig(dt_string + '-fig1.png')


# Figure 2

plt.figure()

df = pd.DataFrame(data={'run': runs, 'step': steps, 'score': scores})
sns.lineplot(x='step', y='score', data=df)

plt.savefig(dt_string + '-fig2.png')


plt.figure()
df = pd.DataFrame(data={'run': runs, 'step': steps, 'harmony': p_harmony})
sns.lineplot(x='step', y='harmony', data=df)

plt.savefig(dt_string + '-fig3.png')


# Figure 4

plt.figure()
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony})

sns.lineplot(x='step', y='probability', hue='parameter', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='probability', var_name='parameter'))

plt.savefig(dt_string + '-fig4.png')


data = {'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony, 'score': scores,
        'n_words': n_words, 'n_user': len(known_indices), 'n_teacher': len(CB_teach), 'n_test': len(CB_test)}

datasaver.save(data, 1)