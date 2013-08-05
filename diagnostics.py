import numpy as np
import matplotlib.pyplot as plt

def print_histogram(data,fname=None):
    np_data = np.asarray(data.values())
    n, bins, patches = plt.hist(np_data, 26, normed=1, facecolor='blue', alpha=0.75)
    if fname!= None:
        plt.title(fname)
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()
    return