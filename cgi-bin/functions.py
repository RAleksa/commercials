import numpy as np

# S(t)
def func_S(t, A, B):
    return A*t + B*np.sin(t)

# z(t)
def func_z(t, A, B):
    return A*t + B*np.cos(t)

# p(t)
def func_p(w, A, B):
    return A*w*(B - w)


def cube_func(x, C, a = 0, b = 1):
    """
    Computes function using matrix of spline coefficients

    Args:
        x(float): function argument
        C(numpy.ndarray - size: N*4 where N=len(func)-1): each row contains 4 coefficients of cubic interpolation
        a, b (float): interval of this function

    Returns:
        float: value of the cubic function of argument x
    """
    step = (b - a) / len(C)

    if x < a:
        return C[0][0] + C[0][1]*x + C[0][2]*x**2 + C[0][3]*x**3

    for n in range(len(C)):
        if a <= x < a + step:
            return C[n][0] + C[n][1]*x + C[n][2]*x**2 + C[n][3]*x**3
        a += step

    if x >= a:
        return C[n][0] + C[n][1]*x + C[n][2]*x**2 + C[n][3]*x**3


