"""
Usage:
    Call function 'interpolate'
    
    Args:
        func_file_name(string): file name of function to interpolate
        interp_coef_file_name(string): file name where to write coefficients of cubic interpolation

    Returns:
        numpy.ndarray(size: N*4 where N=len(func)-1): each row contains 4 coefficients of cubic interpolation
"""

import numpy as np


def interpolate(func_file_name, interp_coef_file_name):
    """
    Interpolates function (cubic spline interpolation)

    Args:
        func_file_name(string): file name of function to interpolate
        interp_coef_file_name(string): file name where to write coefficients of cubic interpolation

    Returns:
        numpy.ndarray(size: N*4 where N=len(func)-1): each row contains 4 coefficients of cubic interpolation
    """
    func_file = open(func_file_name, 'r')

    f = []
    for line in func_file:
        x, y = line.split()
        f.append((float(x), float(y)))
    
    func_file.close()

    interp_coef_file = open(interp_coef_file_name, 'w')
    interp_coef = interpolate_coef(f)

    N = len(f) - 1
    for i in range(N):
        for j in range(4):
            interp_coef_file.write(str(interp_coef[i, j]) + ' ')
        interp_coef_file.write('\n')

    interp_coef_file.close()
    return interp_coef


def sle_gauss(A, b):
    """
    Solves system of linear equations: A*x=b
    
    Args:
        A (numpy.ndarray)
        b (numpy.ndarray)
    
    Returns:
        x (numpy.ndarray)
    """
    X = np.hstack((A, np.split(b, len(b)))).astype(float)

    # Forward Elimination
    for step in range(len(X)):
        maxi = np.argmax(X[step:, step]) + step
        X[step], X[maxi] = X[maxi].copy(), X[step].copy()
        
        X[step] = X[step] / X[step][step]

        for i in range(step + 1, len(X)):
            X[i] -= X[step] * X[i][step]
        
    
    # Back Substitution
    for i in range(len(X) - 2, -1, -1):
        for j in range(i + 1, len(X)):
            X[i] -= X[j] * X[i, j]

    return X[:, -1]


def interpolate_coef(func):
    """
    Real interpolation of function (cubic spline interpolation)

    Args:
        func(list): list of pairs (func argument, func value)

    Returns:
        numpy.ndarray(size: N*4 where N=len(func)-1): each row contains 4 coefficients of cubic interpolation
    """
    N = len(func) - 1
    
    tau = np.zeros(N)
    for i in range(N):
        tau[i] = func[i + 1][0] - func[i][0]
    
    T = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        T[i][i] = (tau[i] + tau[i + 1]) / 3
    for i in range(N - 2):
        T[i + 1][i] = (tau[i + 1]) / 6
        T[i][i + 1] = (tau[i + 1]) / 6
    
    F = np.zeros(N - 1)
    for i in range(N - 1):
        F[i] = (func[i + 2][1] - func[i + 1][1]) / tau[i + 1] - (func[i + 1][1] - func[i][1]) / tau[i]
    
    M = sle_gauss(T, F)
    M = np.hstack((np.asarray([0]), M, np.asarray([0])))
    
    A = np.zeros(N)
    B = np.zeros(N)
    for i in range(N):
        A[i] = (func[i + 1][1] - func[i][1]) / tau[i] - tau[i] / 6 * (M[i + 1] - M[i])
        B[i] = func[i][1] - M[i] * tau[i]**2 / 6 - A[i] * func[i][0]
    
    C = np.zeros((N, 4))
    for i in range(N):
        C[i][0] = (M[i] * func[i + 1][0]**3 - M[i + 1] * func[i][0]**3) / tau[i] / 6 + B[i]
        C[i][1] = (3 * func[i][0]**2 * M[i + 1] - 3 * func[i + 1][0]**2 * M[i]) / tau[i] / 6 + A[i]
        C[i][2] = (-3 * func[i][0] * M[i + 1] + 3 * func[i + 1][0] * M[i]) / tau[i] / 6
        C[i][3] = (-M[i] + M[i + 1]) / tau[i] / 6
    
    return C

