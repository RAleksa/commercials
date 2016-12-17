def tabulate(func, args, segm_numb, file_name, a = 0, b = 1):
    """
    Tabulates function

    Args:
        func: function to tabulate
        args: arguments of function
        segm_number(int): number of segments
        file_name(string): file to save tabulated function

    Returns:
        None
    """
    f = open(file_name, 'w')
    x = a
    n = 0
    while n <= segm_numb:
        f.write(str(x) + ' ' + str(func(x, *args)) + '\n')
        x += (b - a) / segm_numb
        n += 1
    f.close()

