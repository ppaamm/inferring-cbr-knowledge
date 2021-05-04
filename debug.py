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



distance_user = distances_def[2]
harmony_user = False


# Initialization of the test:

print("Initialization of the test")

n_test = 2
CB_test = type48.sample(n=n_test)

#CB_test = pd.concat([type2.sample(n=n_test),
#                       type11.sample(n=n_test),
#                       type38.sample(n=n_test),
#                       type41.sample(n=n_test)])

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

n_runs = 5

for r in range(n_runs):
    print(r)
    
    
    ###########################################################################
    print("Creation of the user")

    n_user = 20
    
    # CB contains all the possible cases that the user could know
#    CB = pd.concat([type2.sample(n=n_user),
#                    type11.sample(n=n_user),
#                    type38.sample(n=n_user),
#                    type41.sample(n=n_user)])
        
    CB = type48.sample(n=n_user)    
    CB = CB.filter(items = ['nominative','inessive']).values.tolist()  
    X = [z[0] for z in CB]
    Y = [z[1] for z in CB]
    n_words = len(CB)
    
    # TODO: Choice of indices
    n_known = 5
    known_indices = list(np.random.permutation(n_user))[:n_known]
    
    # CB_user contains the cases that the user actually knows
    CB_user = [CB[i] for i in known_indices]
    
    X_user = [z[0] for z in CB_user]
    Y_user = [z[1] for z in CB_user]
        
        
    ###########################################################################
    # Teacher's CB
    
    print("Initialisation of the teaching corpus")

    n_teach = 20
#    CB_teach = pd.concat([type2.sample(n=n_teach),
#                          type11.sample(n=n_teach),
#                          type38.sample(n=n_teach),
#                          type41.sample(n=n_teach)])
    
#    CB_teach = CB_teach.filter(items = ['nominative','genitive']).values.tolist()
    CB_teach = type48.sample(n=n_teach)    
    CB_teach = CB_teach.filter(items = ['nominative','inessive']).values.tolist()  
    
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
    
    
    print("Teaching")
    
    for i in range(n_words_teach):
        #print("word", i)

        
        x = CB_teach[randomized[i]][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        result = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_words, distances_def, dict_X, a_solutions, a_orders)
        
        if result == "toto":
            azerty = input("enter")
        
        probas_cb, probas_dist, proba_harmony = result
        
        runs.append(r)
        steps.append(i+1)        
        p_harmony.append(proba_harmony)
        p_d0.append(probas_dist[0])
        p_d1.append(probas_dist[1])
        p_d2.append(probas_dist[2])
        
    data = {'run': runs, 'step': steps, 'd0': p_d0, 'd1': p_d1, 'd2': p_d2, 'harmony': p_harmony, 'score': scores,
        'n_words': n_words, 'n_user': len(known_indices), 'n_teacher': len(CB_teach), 'n_test': len(CB_test)}
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

print('CB:', n_words)
print('User:', len(known_indices))
print('Teacher:', len(CB_teach))
print('Test:', len(CB_test))

datasaver.save(data, 1)