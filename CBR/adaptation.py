from . analogy import solveAnalogy

def adaptation(A, B, C, harmony):
    D = solveAnalogy(A, B, C)[0][0][0]
    return apply_harmony(C, D, harmony)


def apply_harmony(C, D, harmony):
    if harmony:
        if "a" in C or "o" in C or "u" in C:
            D = D.replace("ä", "a")
            D = D.replace("ö", "o")
            D = D.replace("y", "u")
        else:
            D = D.replace("a", "ä")
            D = D.replace("o", "ö")
            D = D.replace("u", "y")
    return D    