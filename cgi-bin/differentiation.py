import numpy as np

def derivative(C):
    """
    Finds derivative of function that consists of cubic polynoms

    Args:
        C(numpy.ndarray - size: N*4 where N=len(func)-1): each row contains 4 coefficients of cubic interpolation

    Returns:
        numpy.ndarray - same size: each row contains 4 coefficients of derivative of the function
    """
    der_C = []
    for c in C:
        der_C.append([c[1], 2*c[2], 3*c[3], 0])

    return np.array(der_C)
