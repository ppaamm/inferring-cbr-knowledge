from CBR import retrieval
from CB_inference import adaptation, update_probas_full, init, evaluate, compare_probas
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt
import datasaver
import datetime

from data.dataloader import DataLoader


###############################################################################
# The user has a fixed CB. 
# The agent infers the content of the CB + distance + vowel harmony
# Retrieval is using 1-NN
###############################################################################




###############################################################################
#### Pipeline

# Data generation:

start_time = time.time()

print("Loading data")
data = DataLoader("./data/FI/Inessive/ine.txt", "inessive")


# Initialization of the test CB:

print("Initialization of the test")

n_test = 100

# Composition of the test CB
CB_test_composition = [(48, n_test)]

CB_test, X_test, Y_test = data.generate_CB(CB_test_composition)





# Parameters: Distances

distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]


# User definition:
distance_user = distances_def[2]
harmony_user = False





###############################################################################



print("Starting the teaching")


runs = []
steps = []
p_harmony = []
p_d0 = []
p_d1 = []
p_d2 = []
scores = []
proba_diff = []

n_runs = 20

for r in range(n_runs):
    print(r)
    
    
    ###########################################################################
    print("Creation of the user")

    n_user = 100
    CB_composition = [(48, n_user)]

    CB, X, Y = data.generate_CB(CB_composition)
    n_words = len(CB)
    
    
    
    # TODO: Choice of indices
    n_known = 30
    known_indices = list(np.random.permutation(n_user))[:n_known]
    
    # CB_user contains the cases that the user actually knows
    CB_user = [CB[i] for i in known_indices]
    X_user = [z[0] for z in CB_user]
    Y_user = [z[1] for z in CB_user]
    
    probas_cb_user = np.array([1 if i in known_indices else 0 for i in range(n_words)])

    ###########################################################################
    # Testing the user
    
    print("Evaluation of the user")
    
    a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)
    idx_distance_user = distances_def.index(distance_user)

    # Evaluating the user on the test base 
    
    Y_test_user = []
    for ii, x in enumerate(X_test):
        NN = a_orders_test[idx_distance_user][ii][0]
        candidate_solutions = [a_solutions_test[harmony_user][ii][n] for n in NN]
        l_sol = [(x,candidate_solutions.count(x)) for x in set(candidate_solutions)]
        Y_test_user.append(l_sol)
        
    
    probas_cb_user = np.array([1 if i in known_indices else 0 for i in range(n_words)])
    probas_dist_user = np.array([0,0,1])
    proba_harmony_user = 0
        
    
    a_solutions_test, a_distances_test, a_orders_test = init(X, Y, X_test, distances_def)
    
    
    ###########################################################################
    # Teacher's CB
    
    print("Initialisation of the teaching corpus")

    n_teach = 50
    CB_teach_composition = [(48, n_teach)]
    CB_teach, X_teach, Y_teach = data.generate_CB(CB_teach_composition)
    
    n_words_teach = len(CB_teach)
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
    proba_diff.append(compare_probas(probas_cb, probas_cb_user))
    
    
    print("Teaching")
    
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
        proba_diff.append(compare_probas(probas_cb, probas_cb_user))
        scores.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))
        
    data = {'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony, 'score': scores,
        'n_words': n_words, 'n_user': len(known_indices), 'n_teacher': len(CB_teach), 'n_test': len(CB_test),
        'p_words': probas_cb, 'X_teach': X_teach, 'CB_user': CB_user}
    datasaver.save(data, '1-Temp' + str(r))

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


# Figure 3

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



# Figure 5

plt.figure()
df = pd.DataFrame(data={'run': runs, 'step': steps, 'proba_diff': proba_diff})
sns.lineplot(x='step', y='proba_diff', data=df)

plt.savefig(dt_string + '-fig5.png')

print('CB:', n_words)
print('User:', len(known_indices))
print('Teacher:', len(CB_teach))
print('Test:', len(CB_test))

datasaver.save(data, 1)