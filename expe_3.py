from CBR import retrieval
from CB_inference import adaptation, update_probas_full, init, evaluate, probabilistic_state_transition, compare_probas
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import datetime
import datasaver

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

df = pd.read_csv ("./data/FI/Inessive/ine.txt")
df = df[df.inessive != '—']
df = df[df.inessive != '–']


#type2 = df[df.type == 2]
#type11 = df[df.type == 11]
#type38 = df[df.type == 38]
#type41 = df[df.type == 41]
type48 = df[df.type == 48]


# Parameters

distances_def = [retrieval.dist2, retrieval.dist3, retrieval.dist5]


# User definition:

print("Creation of the user")


# CB_user contains the cases that the user actually knows
CB_user = []

X_user = []
Y_user = []

distance_user = distances_def[2]
harmony_user = False

transition_proba_estimated = .2   # Transition proba used by the teacher
transition_proba_estimated_2 = .6
        


def test_user(X_test, idx_distance_user, a_orders_test, a_solutions_test, harmony_user):
    Y_test_user = []
    for ii, x in enumerate(X_test):
        NN = a_orders_test[idx_distance_user][ii][0]
        candidate_solutions = [a_solutions_test[harmony_user][ii][n] for n in NN]
        l_sol = [(x,candidate_solutions.count(x)) for x in set(candidate_solutions)]
        Y_test_user.append(l_sol)
    return Y_test_user


# Initialization of the test:

print("Initialization of the test")

n_test = 3
CB_test = type48.sample(n=n_test)
CB_test = CB_test.filter(items = ['nominative','inessive']).values.tolist()


X_test = [z[0] for z in CB_test]
Y_test = [z[1] for z in CB_test]

#a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)


# Teaching



print("Starting the teaching")


runs = []
steps = []
p_harmony = []
p_d0 = []
p_d1 = []
p_d2 = []
scores = []
proba_diff = []

p_harmony_2 = []
p_d0_2 = []
p_d1_2 = []
p_d2_2 = []
scores_2 = []
proba_diff_2 = []

n_runs = 10

for r in range(n_runs):
    print(r)
    
    print("Initialisation of the teaching corpus")

    n_teach = 10
    CB_teach = type48.sample(n=n_teach)    
    CB_teach = CB_teach.filter(items = ['nominative','inessive']).values.tolist()  
    
    n_words_teach = len(CB_teach)
    X_teach = [z[0] for z in CB_teach]
    Y_teach = [z[1] for z in CB_teach]
    dict_X = {X_teach[i]:i for i in range(len(X_teach))}
    
    a_solutions, a_distances, a_orders = init(X_teach, Y_teach, X_teach, distances_def)
    
    
    # used for testing:
    a_solutions_test, a_distances_test, a_orders_test = init(X_teach, Y_teach, X_test, distances_def)

    
    # Priors

    probas_cb = np.zeros(len(CB_teach))
    probas_dist = np.ones(len(distances_def)) / len(distances_def)
    proba_harmony = .5
    
    probas_cb_2 = np.zeros(len(CB_teach))
    probas_dist_2 = np.ones(len(distances_def)) / len(distances_def)
    proba_harmony_2 = .5
    

    runs.append(r)
    steps.append(0)
    p_harmony.append(proba_harmony)
    p_d0.append(probas_dist[0])
    p_d1.append(probas_dist[1])
    p_d2.append(probas_dist[2])
    scores.append(1.)
    proba_diff.append(0)
    
    p_harmony_2.append(proba_harmony)
    p_d0_2.append(probas_dist[0])
    p_d1_2.append(probas_dist[1])
    p_d2_2.append(probas_dist[2])
    scores_2.append(1.)
    proba_diff_2.append(0)
    
    # Case 0: necessarily retained
    
    
    probas_cb_user = np.zeros(len(CB_teach))
    CB_user = [CB_teach[0]]
    X_user = [CB_teach[0][0]]
    Y_user = [CB_teach[0][1]]
    
    idx_distance_user = distances_def.index(distance_user)
    probas_cb[dict_X[CB_teach[0][0]]] = 1.
    probas_cb_2[dict_X[CB_teach[0][0]]] = 1.
    probas_cb_user[dict_X[CB_teach[0][0]]] = 1.
    
    
    Y_test = test_user(X_test, idx_distance_user, a_orders_test, a_solutions_test, harmony_user)
    
    
    for i in range(1,n_words_teach):
        x = CB_teach[i][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        probas_cb, probas_dist, proba_harmony = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X_teach, Y_teach, n_words_teach, distances_def, dict_X, a_solutions, a_orders)
        probas_cb_2, probas_dist_2, proba_harmony_2 = update_probas_full(x, y, probas_cb_2, probas_dist_2, proba_harmony_2, X_teach, Y_teach, n_words_teach, distances_def, dict_X, a_solutions, a_orders)
        
        # Update
        probas_cb = probabilistic_state_transition(x, probas_cb, dict_X, transition_proba_estimated)
        probas_cb_2 = probabilistic_state_transition(x, probas_cb_2, dict_X, transition_proba_estimated_2)
        
        
        
        ###################################
        #       RETENTION MODEL           #
        ###################################
        
        success = (y == CB_teach[i][1])
        if (success and np.random.rand() < .5) or (not(success) and np.random.rand() < .8):
            CB_user.append(CB_teach[i])
            X_user.append(CB_teach[i][0])
            Y_user.append(CB_teach[i][1])
            probas_cb_user[dict_X[CB_teach[i][0]]] = 1.
        
        ###################################
        ###################################
        
        
        
        
        runs.append(r)
        steps.append(i+1)        
        p_harmony.append(proba_harmony)
        p_d0.append(probas_dist[0])
        p_d1.append(probas_dist[1])
        p_d2.append(probas_dist[2])
        proba_diff.append(compare_probas(probas_cb, probas_cb_user))
        
        p_harmony_2.append(proba_harmony_2)
        p_d0_2.append(probas_dist_2[0])
        p_d1_2.append(probas_dist_2[1])
        p_d2_2.append(probas_dist_2[2])
        proba_diff_2.append(compare_probas(probas_cb_2, probas_cb_user))
        
        a_solutions_test, a_distances_test, a_orders_test = init(X_user, Y_user, X_test, distances_def)
        Y_test_user = test_user(X_test, idx_distance_user, a_orders_test, a_solutions_test, harmony_user)
        scores.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb, probas_dist, proba_harmony))
        scores_2.append(evaluate(X_test, Y_test_user, a_solutions_test, a_distances_test, a_orders_test, CB_user, distances_def, probas_cb_2, probas_dist_2, proba_harmony_2))
        

# Plot


print("--- %s seconds ---" % (time.time() - start_time))
        
dt_string = 'expe3-' +  datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


# Figure 1

plt.figure()
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2})

sns.lineplot(x='step', y='probability', hue='distance', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='probability', var_name='distance'))

plt.savefig(dt_string + '-fig1.png')


# Figure 1 bis

plt.figure()
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 'd0': p_d0_2, 'd1': p_d1_2, 'd2': p_d2_2})

sns.lineplot(x='step', y='probability', hue='distance', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='probability', var_name='distance'))

plt.savefig(dt_string + '-fig1_2.png')


# Figure 1 ter



plt.figure()
        
df = pd.DataFrame(data={'run': runs, 'step': steps, 
                        'd0 (teacher 1)': p_d0, 'd0 (teacher 2)': p_d0_2,
                        'd1 (teacher 1)': p_d1, 'd1 (teacher 2)': p_d1_2, 
                        'd2 (teacher 1)': p_d2, 'd2 (teacher 2)': p_d2_2})

sns.lineplot(x='step', y='probability', hue='distance', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='probability', var_name='distance'))

plt.savefig(dt_string + '-fig1_3.png')





# Figure 2

plt.figure()

df = pd.DataFrame(data={'run': runs, 'step': steps, 'Teacher 1': scores, 'Teacher 2': scores_2})
sns.lineplot(x='step', y='score', hue='Teacher', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='score', var_name='Teacher'))

plt.savefig(dt_string + '-fig2.png')


# Figure 3

plt.figure()
df = pd.DataFrame(data={'run': runs, 'step': steps, 'Teacher 1': p_harmony, 'Teacher 2': p_harmony_2})
sns.lineplot(x='step', y='harmony', hue='Teacher', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='harmony', var_name='Teacher'))

plt.savefig(dt_string + '-fig3.png')


# Figure 5

plt.figure()

df = pd.DataFrame(data={'run': runs, 'step': steps, 'Teacher 1': proba_diff, 'Teacher 2': proba_diff_2})
sns.lineplot(x='step', y='proba_diff', hue='Teacher', 
             data=pd.melt(df, id_vars = ['step', 'run'], value_name='proba_diff', var_name='Teacher'))


plt.savefig(dt_string + '-fig5.png')



data = {'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony, 'score': scores,
        'n_teacher': len(CB_teach), 'n_test': len(CB_test)}

datasaver.save(data, 3)