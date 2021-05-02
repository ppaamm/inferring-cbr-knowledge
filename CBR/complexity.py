from . utils import getTransformationPart1, getLengthLetter, getLengthInstruction, getTransformation2
import sys

# Used to compute complexities

complexity_buffer = {}


def getK_A(a):
    """
    returns K(A)
    """
    
    if a in complexity_buffer: 
        #print('using buffer')
        return complexity_buffer[a]
    
    result_transf_1 = []
    result_varA = []

    getTransformationPart1("", a, a, [], [], result_transf_1, result_varA, [])
    
    min_dist = len(a) * getLengthLetter()
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x], 1, 0, 0)
        if ll < min_dist: 
            min_dist = ll
            #print(result_transf_1[x], result_varA[x])
    
    
    complexity_buffer[a] = min_dist
    return min_dist


def getK_AB(a,b):
    ab = a + ':' + b
    if ab in complexity_buffer: return complexity_buffer[ab]
    
    # Compute K(A:B)
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
    
    complexity_buffer[ab] = min_length
    return min_length


def getK_AC(a,c):
    ac = a + '::' + c
    if ac in complexity_buffer: return complexity_buffer[ac]
    
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
            
    complexity_buffer[ac] = min_dist
    return min_dist
    