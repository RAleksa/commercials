def integrate(func_file_name, a = 0, b = 1):
    """
    Integrates function

    Args:
        func_file_name(string): file name of function to integrate
        a, b (float): interval of integration

    Returns:
        float: integral of function
    """
    func_file = open(func_file_name, 'r')

    func = []
    for line in func_file:
        x, y = line.split()
        x, y = float(x), float(y)
        if a <= x <= b:
            func.append((x, y))
    
    func_file.close()
    return integrate_list(func)


def integrate_list(func):
    """
    Real integrate function (rectangle method)

    Args:
        func(list): list of pairs (func argument, func value)

    Returns:
        float: integral of function
    """
    integral = 0
    for i in range(1, len(func)):
        integral += (func[i][0] - func[i - 1][0]) * (func[i][1] + func[i - 1][1]) / 2
    return integral
