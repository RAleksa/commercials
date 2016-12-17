def cauchy(f1, f2, x10, x20, T, h):
    """
    Solves a Cauchy problem using Runge–Kutta fourth-order method for system of two equations

    Args:
        f1, f2 (functions): rigth parts of Cauchy system of two equations with arguments: t, x1, x2
        x10, x20: initial conditions x1(0), x2(0)
        T (float): maximum value of x
        h (int): number of segments

    Return:
        x1, x2 (list of floats): tabulated solution
    """
    x1 = [x10]
    x2 = [x20]

    for i in range(h):
        k11 = f1(i*T/h, x1[-1], x2[-1])
        k12 = f2(i*T/h, x1[-1], x2[-1])
        k21 = f1(i*T/h + T/h/2, x1[-1] + T/h/2*k11, x2[-1] + T/h/2*k12)
        k22 = f2(i*T/h + T/h/2, x1[-1] + T/h/2*k11, x2[-1] + T/h/2*k12)
        k31 = f1(i*T/h + T/h/2, x1[-1] + T/h/2*k21, x2[-1] + T/h/2*k22)
        k32 = f2(i*T/h + T/h/2, x1[-1] + T/h/2*k21, x2[-1] + T/h/2*k22)
        k41 = f1(i*T/h + T/h, x1[-1] + T/h*k31, x2[-1] + T/h*k32)
        k42 = f2(i*T/h + T/h, x1[-1] + T/h*k31, x2[-1] + T/h*k32)
        
        x1.append(x1[-1] + T/h/6*(k11 + 2*k21 + 2*k31 + k41))
        x2.append(x2[-1] + T/h/6*(k12 + 2*k22 + 2*k32 + k42))
    
    return x1, x2

