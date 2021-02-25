from . analogy import solveAnalogy
from . utils import getTransformationPart1, getLengthLetter, getLengthInstruction, getTransformation2
import sys


def retrieval(CB, C, dist):
    min_dist = sys.maxsize
    retrieved_case = []
    for ab in CB:
        distance = dist(ab, C)
        if distance == min_dist: retrieved_case.append(ab)
        elif distance < min_dist: 
            min_dist = distance
            retrieved_case = [ab]
    return retrieved_case, min_dist


def retrieval_k(CB, C, dist, k):
    distances = [[ab, dist(ab,C)] for ab in CB]
    distances.sort(key=lambda x: x[1])
    return distances[:k]
    


def dist1(ab,c):
    """
    d1(A:B,C) = min_D K(A:B::C:D)
    """
    a = ab[0]
    b = ab[1]
    _, dist = solveAnalogy(a,b,c)
    return dist
    

def getK_A(a):
    """
    returns K(A)
    """
    result_transf_1 = []
    result_varA = []

    getTransformationPart1("", a, a, [], [], result_transf_1, result_varA, [])
    
    min_dist = len(a) * getLengthLetter()
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x], 1, 0, 0)
        if ll < min_dist: 
            min_dist = ll
            #print(result_transf_1[x], result_varA[x])

    return min_dist

def dist2(ab,c):
    """
    d2(A:B,C) = min_D K(A:B::C:D) - K(A)
    """
    a = ab[0]
    b = ab[1]
    _, dist = solveAnalogy(a,b,c)
    #return dist - len(a) * length_letter
    return dist - getK_A(a)

def dist3(ab,c):
    """
    d3(A:B,C) = min_D K(A:B::C:D) - K(A::B)
    """
    a = ab[0]
    b = ab[1]
    _, dist = solveAnalogy(a,b,c)
    
    # Compute K(A::B)
    result_transf_1 = []
    result_varA = []


    getTransformationPart1("", a, a, [], [], result_transf_1, result_varA, [])
    min_length = sys.maxsize
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x], 1, 1, 0)
        if (ll <= min_length):
            result_transf_2 = []
            result_varB = []
            l = result_varA[x]
            getTransformation2(result_transf_1[x] + ",:", b, l, result_transf_2, result_varB)
            #print(result_transf_2, result_varB)
            for y in range(len(result_transf_2)):
                ll = getLengthInstruction(result_transf_2[y], result_varB[y], 1, 1, 0)
                if ll < min_length: 
                    #print("--------")
                    #print(result_transf_2[y])
                    #print(result_varB[y])
                    #print('--------')
                    min_length = ll
    
    return dist - min_length


def dist4(ab,c):
    """
    d4(A:B,C) = K(A::C)
    """
    a = ab[0]
    
    transformations = []
    varA = []
    varC = []

    getTransformationPart1("", a, c, [], [], transformations, varA, varC)
    
    min_dist = sys.maxsize
    
    for x in range(len(transformations)):
        ll = getLengthInstruction(transformations[x], varA[x] + varC[x], 2, 0, 1)
        if ll < min_dist: 
            min_dist = ll
            # print(transformations[x])
    return min_dist


def dist5(ab,c):
    """
    d5(A:B,C) = K(A::C) - K(A)
    """
    #return dist4(ab,c) - len(ab[0]) * length_letter
    return dist4(ab,c) - getK_A(ab[0])


###############################################################################
    
def compare_distances(a,b,c):
    ab=[a,b]
    print(dist1(ab,c), dist2(ab,c), dist3(ab,c), dist4(ab,c), dist5(ab,c))