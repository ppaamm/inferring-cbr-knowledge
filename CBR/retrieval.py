from . analogy import solveAnalogy
from . import complexity
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
    



def dist2(ab,c):
    """
    d2(A:B,C) = min_D K(A:B::C:D) - K(A)
    """
    a = ab[0]
    b = ab[1]
    _, dist = solveAnalogy(a,b,c)
    #return dist - len(a) * length_letter
    return dist - complexity.getK_A(a)

def dist3(ab,c):
    """
    d3(A:B,C) = min_D K(A:B::C:D) - K(A:B)
    """
    a = ab[0]
    b = ab[1]
    _, dist = solveAnalogy(a,b,c)
    
    min_length = complexity.getK_AB(a,b)
    
    return dist - min_length


def dist4(ab,c):
    """
    d4(A:B,C) = K(A::C)
    """
    a = ab[0]
    return complexity.getK_AC(a,c)


def dist5(ab,c):
    """
    d5(A:B,C) = K(A::C) - K(A)
    """
    #return dist4(ab,c) - len(ab[0]) * length_letter
    return dist4(ab,c) - complexity.getK_A(ab[0])


###############################################################################
    
def compare_distances(a,b,c):
    ab=[a,b]
    print(dist1(ab,c), dist2(ab,c), dist3(ab,c), dist4(ab,c), dist5(ab,c))