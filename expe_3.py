from CBR import retrieval
from CB_inference import adaptation, update_probas_full, init, evaluate, probabilistic_state_transition
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt

###############################################################################
# The user learns during the process
# The agent infers the content of the CB + distance + vowel harmony
# Retrieval is using 1-NN
###############################################################################



###############################################################################
#### Pipeline

# Data generation:

start_time = time.time()

print("Loading data")

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

print("Creation of the user")


# CB_user contains the cases that the user actually knows
CB_user = []

X_user = []
Y_user = []

distance_user = distances_def[2]
harmony_user = True
transition_proba_estimated = .2   # Transition proba used by the teacher

def retention_user(i, success):
    if (success and np.random.rand() < .5) or (not(success) and np.random.rand() < .8):
        CB_user.append(CB_teach[i])
        X_user.append(CB_teach[i][0])
        Y_user.append(CB_teach[i][1])
        


# Initialization of the test:

print("Initialization of the test")

n_test = 10

CB_test = pd.concat([type2.sample(n=n_test),
                       type11.sample(n=n_test),
                       type38.sample(n=n_test),
                       type41.sample(n=n_test)])

CB_test = CB_test.filter(items = ['nominative','genitive']).values.tolist()




X_test = [z[0] for z in CB_test]
Y_test = [z[1] for z in CB_test]

#a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)



# Evaluating the user on the test base

#print("Evaluation of the user")
#
#Y_test_user = []
#for x in X_test:
#    source, _ = retrieval.retrieval(CB_user, x, distance_user)
#    Y_test_user.append(adaptation(source[0][0], source[0][1], x, harmony_user))



# Teaching


print("Initialisation of the teaching corpus")

n_teach = 4
CB_teach = pd.concat([type2.sample(n=n_teach),
                      type11.sample(n=n_teach),
                      type38.sample(n=n_teach),
                      type41.sample(n=n_teach)])

CB_teach = CB_teach.filter(items = ['nominative','genitive']).values.tolist()
n_words_teach = len(CB_teach)
X_teach = [z[0] for z in CB_teach]
Y_teach = [z[1] for z in CB_teach]
dict_X = {X_teach[i]:i for i in range(len(X_teach))}

a_solutions, a_distances, a_orders = init(X_teach, Y_teach, X_teach, distances_def)

# used for testing:
a_solutions_test, a_distances_test, a_orders_test = init(X_teach, Y_teach, X_test, distances_def)


print("Starting the teaching")


runs = []
steps = []
p_harmony = []
p_d0 = []
p_d1 = []
p_d2 = []
scores = []

n_runs = 20

for r in range(n_runs):
    print(r)
    
    # Priors

    probas_cb = np.zeros(len(CB_teach))
    probas_dist = np.ones(len(distances_def)) / len(distances_def)
    proba_harmony = .5
    
    
    randomized = random.sample(list(range(n_words_teach)), n_words_teach)
    runs.append(r)
    steps.append(0)
    p_harmony.append(proba_harmony)
    p_d0.append(probas_dist[0])
    p_d1.append(probas_dist[1])
    p_d2.append(probas_dist[2])
    scores.append(1.)
    
    # Case 0: necessarily retained
    
    CB_user = [CB_teach[randomized[0]]]
    X_user = [CB_teach[randomized[0]][0]]
    Y_user = [CB_teach[randomized[0]][1]]
    
    
    probas_cb[dict_X[CB_teach[randomized[0]][0]]] = 1.
    
    
    for i in range(1,n_words_teach):
        #print("word", i)

        
        x = CB_teach[randomized[i]][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        probas_cb, probas_dist, proba_harmony = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X_teach, Y_teach, n_words_teach, distances_def, dict_X, a_solutions, a_orders)
        
        # Update
        probas_cb = probabilistic_state_transition(x, probas_cb, dict_X, transition_proba_estimated)
        
        retention_user(randomized[i], y == CB_teach[randomized[i]][1])
        
        runs.append(r)
        steps.append(i+1)        
        p_harmony.append(proba_harmony)
        p_d0.append(probas_dist[0])
        p_d1.append(probas_dist[1])
        p_d2.append(probas_dist[2])
        
        Y_test_user = []
        for x in X_test:
            source, _ = retrieval.retrieval(CB_user, x, distance_user)
            Y_test_user.append(adaptation(source[0][0], source[0][1], x, harmony_user))
        
        
        scores.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))


# Plot


print("--- %s seconds ---" % (time.time() - start_time))
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'p_d0': p_d0, 'p_d1': p_d1, 'p_d2': p_d2})

sns.lineplot(x='step', y='value', hue='variable', 
             data=pd.melt(df, id_vars = ['step', 'run']))

plt.figure()

df = pd.DataFrame(data={'run': runs, 'step': steps, 'score': scores})
sns.lineplot(x='step', y='score', data=df)

