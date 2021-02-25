import sys
from . utils import getTransformationPart1, getLengthInstruction, getPart2, applyTransformation, writeInstruction, getTransformation2



def solveAnalogy(A, B, C):
    """
    Returns a solution (or a list of valid solutions) for 
    the analogical equation "`A` : `B` # `C` : x", where 
    each solution is constituted of the solution term D 
    and the corresponding transformation.
    """
    
    min_length_result = len(C) + len(B) - len(A)

    final_result = []

    result_transf_1 = []
    result_varA = []
    result_varC = []
    list_varA = []
    list_varC = []

    getTransformationPart1("", A, C, list_varA, list_varC, result_transf_1, result_varA, result_varC)
    min_length = sys.maxsize
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x] + result_varC[x])
        if (ll <= min_length):
            result_transf_2 = []
            result_varB = []
            l = result_varA[x]
            getTransformation2(result_transf_1[x] + ",:", B, l, result_transf_2, result_varB)
            for y in range(len(result_transf_2)):

                ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varC[x])
                if (ll <= min_length):
                    partInstruction_B = getPart2(result_transf_2[y])
                    result_varD = list(result_varC[x])
                    D = applyTransformation(partInstruction_B, result_varD)
                    ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varD)

                    if (ll < min_length and len(D) >= min_length_result):
                        min_length = ll
                        final_result = [ [D, writeInstruction(result_transf_2[y], result_varB[y], result_varD)] ]
                    elif (ll == min_length and len(D) >= min_length_result):
                        final_result.append([D, writeInstruction(result_transf_2[y], result_varB[y], result_varD)])
    return final_result, min_length






def solveAnalogy_proba(A, B, C):
    """
    Returns a solution (or a list of valid solutions) for 
    the analogical equation "`A` : `B` # `C` : x", where 
    each solution is constituted of the solution term D 
    and the corresponding transformation.
    """
    
    possible_results = {}

    result_transf_1 = []
    result_varA = []
    result_varC = []
    list_varA = []
    list_varC = []

    getTransformationPart1("", A, C, list_varA, list_varC, result_transf_1, result_varA, result_varC)
    min_length = sys.maxsize
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x] + result_varC[x])
        if (ll <= min_length):
            result_transf_2 = []
            result_varB = []
            l = result_varA[x]
            getTransformation2(result_transf_1[x] + ",:", B, l, result_transf_2, result_varB)
            for y in range(len(result_transf_2)):

                ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varC[x])
                if (ll <= min_length):
                    partInstruction_B = getPart2(result_transf_2[y])
                    result_varD = list(result_varC[x])
                    D = applyTransformation(partInstruction_B, result_varD)
                    ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varD)
                    
                    if D in possible_results:
                        possible_results[D] += 2**(-ll)
                    else: 
                        possible_results[D] = 2 **(-ll)
                        
    # Normalization
    factor = 1.0/sum(possible_results.values())
    for D in possible_results: possible_results[D] *= factor
    return dict(sorted(possible_results.items(), key=lambda item: item[1], reverse=True))