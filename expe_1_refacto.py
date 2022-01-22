from CBR import retrieval
from CB_inference import adaptation, evaluate, compare_probas
from CB_inference import PreComputation, InferenceEngine
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
data_loader = DataLoader("./data/FI/Inessive/ine.txt", "nominative", "inessive")


# Initialization of the test CB:

print("Initialization of the test")

n_test = 3

# Composition of the test CB
CB_test_composition = [(48, n_test)]

CB_test, X_test, Y_test = data_loader.generate_CB(CB_test_composition)





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

n_runs = 10

for r in range(n_runs):
    print(r)
    
    
    ###########################################################################
    print("Creation of the user")

    n_user = 5
    CB_composition = [(48, n_user)]

    CB, X, Y = data_loader.generate_CB(CB_composition)
    n_words = len(CB)
    
    
    
    # TODO: Choice of indices
    n_known = 2
    known_indices = list(np.random.permutation(n_user))[:n_known]
    
    # CB_user contains the cases that the user actually knows
    CB_user = [CB[i] for i in known_indices]
    X_user = [z[0] for z in CB_user]
    Y_user = [z[1] for z in CB_user]
    
    probas_cb_user = np.array([1 if i in known_indices else 0 for i in range(n_words)])

    ###########################################################################
    # Testing the user
    
    print("Evaluation of the user")
    
    user_test_precomp = PreComputation(X_user, Y_user, X_test, distances_def)
    idx_distance_user = distances_def.index(distance_user)

    # Evaluating the user on the test base 
    
    Y_test_user = []
    for ii, x in enumerate(X_test):
        NN = user_test_precomp.a_orders[idx_distance_user][ii][0]
        candidate_solutions = [user_test_precomp.a_solutions[harmony_user][ii][n] for n in NN]
        l_sol = [(x,candidate_solutions.count(x)) for x in set(candidate_solutions)]
        Y_test_user.append(l_sol)
        
    
    probas_cb_user = np.array([1 if i in known_indices else 0 for i in range(n_words)])
    probas_dist_user = np.array([0,0,1])
    proba_harmony_user = 0
        
    
    full_test_precomp = PreComputation(X, Y, X_test, distances_def)
    
    
    ###########################################################################
    # Teacher's CB
    
    print("Initialisation of the teaching corpus")

    n_teach = 3
    CB_teach_composition = [(48, n_teach)]
    CB_teach, X_teach, Y_teach = data_loader.generate_CB(CB_teach_composition)
    
    n_words_teach = len(CB_teach)
    dict_X = {X_teach[i]: i for i in range(len(X_teach))}
    
    teacher_precomputation = PreComputation(X, Y, X_teach, distances_def)
        
    # Priors

    prior_cb = .5 * np.ones(len(CB))
    prior_dist = np.ones(len(distances_def)) / len(distances_def)
    prior_harmony = .5
    
    
    inference = InferenceEngine(CB, prior_cb, prior_dist, prior_harmony, teacher_precomputation)
    
    
    
    randomized = random.sample(list(range(n_words_teach)), n_words_teach)
    runs.append(r)
    steps.append(0)
    p_harmony.append(inference.proba_harmony)
    p_d0.append(inference.probas_dist[0])
    p_d1.append(inference.probas_dist[1])
    p_d2.append(inference.probas_dist[2])
    scores.append(evaluate(X_test, Y_test_user, 
                           full_test_precomp.a_solutions, 
                           full_test_precomp.a_distances, 
                           full_test_precomp.a_orders, 
                           CB_user, 
                           distances_def, 
                           inference.probas_cb, 
                           inference.probas_dist, 
                           inference.proba_harmony))
    proba_diff.append(compare_probas(inference.probas_cb, probas_cb_user))
    
    
    print("Teaching")
    
    for i in range(n_words_teach):
        #print("word", i)

        
        x = CB_teach[randomized[i]][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        inference.update_probas(dict_X[x], y)

        
        runs.append(r)
        steps.append(i+1)        
        p_harmony.append(inference.proba_harmony)
        p_d0.append(inference.probas_dist[0])
        p_d1.append(inference.probas_dist[1])
        p_d2.append(inference.probas_dist[2])
        proba_diff.append(compare_probas(inference.probas_cb, probas_cb_user))
        scores.append(evaluate(X_test, 
                               Y_test_user, 
                               full_test_precomp.a_solutions, 
                               full_test_precomp.a_distances, 
                               full_test_precomp.a_orders, 
                               CB_user, 
                               distances_def, 
                               inference.probas_cb, 
                               inference.probas_dist, 
                               inference.proba_harmony))
        
    data = {'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony, 'score': scores,
        'n_words': n_words, 'n_user': len(known_indices), 'n_teacher': len(CB_teach), 'n_test': len(CB_test),
        'p_words': inference.probas_cb, 'X_teach': X_teach, 'CB_user': CB_user}
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