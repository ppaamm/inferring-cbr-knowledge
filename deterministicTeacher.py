"""
Deterministic teacher: 
- The teacher knows the characteristics of the CBR agent
- The CBR agent is deterministic
- The CBR agent retains all cases
"""

# temp
import random
import string
import time
from CBR import retrieval
from CBR import analogy
# end temp


from CBR import cbragent

class DeterministicTeacher:
    
    def __init__(self, trainBatch, testBatch, distance, analogy, k_neighbors, verbose=False):
        self.name = 'Deterministic'
        self.user = cbragent.CBRAgent(distance, analogy, k_neighbors)
        self.verbose = verbose
        self.trainBatch = trainBatch
        self.testBatch = testBatch
        
    def changeVerbose(self, verbose):
        self.verbose = verbose
    
    def printv(self, message):
        if self.verbose: print(message)
        
    def estimated_score(self, learner = None):
        if learner == None: learner = self.user
        n = len(self.testBatch)
        n_success = .0
        for case in self.testBatch:
            if learner.solve_problem(case[0]) == case[1]: n_success += 1
        return n_success / n
    
    
    
    def playAction(self, case):
        self.printv("Playing case: " + case[0] + " : " + case[1])
        user_response = self.user.solve_problem(case[0])
        self.printv("User's response: " + user_response)
        self.user.retain(case[0], case[1])
        self.printv("User's CB updated")
    
    
    def chooseAction(self, horizon, learner=None):
        if learner == None: learner = self.user.clone()
        
        scores = []
        for case in self.trainBatch:
            # Simulate what happens when displaying this case to the learner
            #print(horizon, case)
            learner_rollout = learner.clone()
            learner_rollout.retain(case[0], case[1])
            if horizon == 0: score = self.estimated_score(learner=learner_rollout)
            else: _, score = self.chooseAction(horizon-1, learner=learner_rollout)
            scores.append([case, score])
        result = max(scores, key=lambda x: x[1])
        
        return result[0], result[1]
        



###############################################################################
## Example        


def get_random_string(max_length):
    # choose from all lowercase letter
    length = random.randint(2, max_length)
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def create_batch(n_a, n_us, n_um=0):
    batch = []
    for _ in range(n_a):
        base = get_random_string(5)
        batch.append([base + 'a', base + 'am'])
    for _ in range(n_us):
        base = get_random_string(5)
        batch.append([base + 'us', base + 'um'])
    for _ in range(n_um):
        base = get_random_string(5)
        batch.append([base + 'um', base + 'um'])    
    return batch
    
    
train_batch = create_batch(2,2,2)
test_batch = create_batch(10,30,20) # more a-type than us-type


teacher = DeterministicTeacher(train_batch, test_batch, retrieval.dist5, analogy.solveAnalogy, 1, verbose=False)


start_time = time.time()

# Learning on two steps
# Step 1:
action, _ = teacher.chooseAction(horizon=2)
teacher.playAction(action)
print('First action:', action)

# Step 2:
action, _ = teacher.chooseAction(horizon=1)
teacher.playAction(action)
print('Second action:', action)

# Step 3:
action, _ = teacher.chooseAction(horizon=0)
teacher.playAction(action)
print('Thrid action:', action)


print("Final score:", teacher.estimated_score())

print("--- %s seconds ---" % (time.time() - start_time))


