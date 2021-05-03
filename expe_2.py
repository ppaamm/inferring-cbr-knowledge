from CBR import retrieval
from CB_inference import adaptation, update_probas_full, init, evaluate, update_probas_states
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt

###############################################################################
# Magic doesn't exist
# The user has a fixed CB. 
# The agent infers the content of the CB + distance + vowel harmony
# Retrieval is using 1-NN
###############################################################################







###############################################################################
#### Variant 1: [["maa"], harmony] vs [["maa", "pää"], no-harmony]


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

n_user = 1

# CB contains all the possible cases that the user could know
CB = [["makkara", "makkarassa"], ["metsä", "metsässä"]]
X = [z[0] for z in CB]
Y = [z[1] for z in CB]
n_words = len(X)

# TODO: Choice of indices
known_indices = [1]

# CB_user contains the cases that the user actually knows
CB_user = [CB[i] for i in known_indices]

X_user = [z[0] for z in CB_user]
Y_user = [z[1] for z in CB_user]

distance_user = distances_def[2]
harmony_user = True




# Teaching


print("Initialisation of the teaching corpus")

CB_teach = [['koira', 'koirassa'],
            ['mäyrä', 'mäyrässä'],
            ['talo', 'talossa'],
            ['piha', 'pihassa'], 
            ['kypärä', 'kypärässä'], 
            ['ryhmä', 'ryhmässä'],
            ['kallio', 'kalliossa'],
            ['löyly', 'löylyssä']]

n_words_teach = len(CB_teach)
X_teach = [z[0] for z in CB_teach]
dict_X = {X_teach[i]:i for i in range(len(X_teach))}

a_solutions, a_distances, a_orders = init(X, Y, X_teach, distances_def)



# Defining the states

states = []

#for d in distances_def:
#    for h in (True, False):
#        states.append({'distance': d,
#                       'harmony': h,
#                       'cases': [0]})
#        states.append({'distance': d,
#                       'harmony': h,
#                       'cases': [1]})
#        states.append({'distance': d,
#                       'harmony': h,
#                       'cases': [0, 1]})


states.append({'distance': 2,
               'harmony': True,
               'cases': [0]})

states.append({'distance': 2,
               'harmony': True,
               'cases': [1]})

states.append({'distance': 2,
               'harmony': False,
               'cases': [0, 1]})


n_states = len(states)
print('Number of states:', n_states)


runs = []
steps = []
p_state_0 = []
p_state_1 = []
p_state_2 = []

n_runs = 50

for r in range(n_runs):
    print(r)
    
    # Priors

    probas_state = np.ones(n_states) / n_states
    
    
    randomized = random.sample(list(range(n_words_teach)), n_words_teach)
    runs.append(r)
    steps.append(0)
    p_state_0.append(probas_state[0])
    p_state_1.append(probas_state[1])
    p_state_2.append(probas_state[2])
    
    for i in range(n_words_teach):
        #print("word", i)

        
        x = CB_teach[randomized[i]][0]
        source, _ = retrieval.retrieval(CB_user, x, distance_user)
        y = adaptation(source[0][0], source[0][1], x, harmony_user)
        probas_state = update_probas_states(x, y, probas_state, X, Y, n_words, states, dict_X, a_solutions, a_orders)
        #probas_cb, probas_dist, proba_harmony = update_probas_full(x, y, probas_cb, probas_dist, proba_harmony, X, Y, n_words, distances_def, dict_X, a_solutions, a_orders)
        
        runs.append(r)
        steps.append(i+1)
        p_state_0.append(probas_state[0])
        p_state_1.append(probas_state[1])
        p_state_2.append(probas_state[2])

# Plot


print("--- %s seconds ---" % (time.time() - start_time))

plt.figure()
plt.title("Probability of (makkara + harmony)")
df = pd.DataFrame(data={'run': runs, 'step': steps, 
                        'p_(makkara)': p_state_0, 'p_(metsä)': p_state_1, 
                        'p_(makkara,metsä)': p_state_2})
sns.lineplot(x='step', y='value', hue='variable', 
             data=pd.melt(df, id_vars = ['step', 'run']))









###############################################################################


def expe2_marginals():
    print("Starting the teaching") 
    
    runs = []
    steps = []
    p_harmony = []
    p_d0 = []
    p_d1 = []
    p_d2 = []
    p_maa = []
    p_paa = []
    
    n_runs = 20
    
    for r in range(n_runs):
        print(r)
        
        # Priors
    
        #probas_cb = .5 * np.ones(len(CB))
        probas_cb = np.array([.8, .2])
        probas_dist = np.ones(len(distances_def)) / len(distances_def)
        proba_harmony = .5
        
        
        randomized = random.sample(list(range(n_words_teach)), n_words_teach)
        runs.append(r)
        steps.append(0)
        p_harmony.append(proba_harmony)
        p_d0.append(probas_dist[0])
        p_d1.append(probas_dist[1])
        p_d2.append(probas_dist[2])
        p_maa.append(probas_cb[0])
        p_paa.append(probas_cb[1])
        
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
            p_maa.append(probas_cb[0])
            p_paa.append(probas_cb[1])
    
    # Plot
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.figure() 
    plt.title("Proba distances")
    
    df = pd.DataFrame(data={'run': runs, 'step': steps, 'p_d0': p_d0, 'p_d1': p_d1, 'p_d2': p_d2})
    sns.lineplot(x='step', y='value', hue='variable', 
                 data=pd.melt(df, id_vars = ['step', 'run']))
    
    plt.figure()
    
    plt.title("Maa / Pää")
    df = pd.DataFrame(data={'run': runs, 'step': steps, 'p_maa': p_maa, 'p_paa': p_paa})
    sns.lineplot(x='step', y='value', hue='variable', 
                 data=pd.melt(df, id_vars = ['step', 'run']))
    
    plt.figure()
    plt.title("Vowel harmony")
    df = pd.DataFrame(data={'run': runs, 'step': steps, 'harmony': p_harmony})
    sns.lineplot(x='step', y='harmony', data=df)



