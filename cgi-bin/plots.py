import matplotlib.pyplot as plt
import numpy as np


def plot(func_file_name, func_id, xlabel, ylabel, a, b):
    """
    Creates a plot of function and saves an image to a file

    Args:
        func_file_name (string): function to show
        func_id (string): just some id to save the plot
        xlabel, ylabel (string): names of x and y axes of the plot
        a, b (float): limits of x axe

    Returs:
        None
    """
    func_file = open(func_file_name, 'r')

    x = []
    y = []
    for line in func_file:
        xi, yi = line.split()
        x.append(float(xi))
        y.append(float(yi))
    
    func_file.close()

    fig = plt.figure()
    plt.plot(x, y, color="green", linewidth=2)
    plt.xlabel(xlabel, fontsize=23)
    plt.ylabel(ylabel, fontsize=23)
    plt.xlim(a, b)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.tight_layout()

    fig.savefig("plot_" + str(func_id) + ".png")

