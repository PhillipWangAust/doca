from doca import Doca
import os
import numpy as np


def ensure_dir(file_path):
    """
    Use this function to make sure a directory exists.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def proc(id):
    output_dir = "results/{}/".format(id)
    ensure_dir(output_dir)

    data = np.loadtxt(fname="data/loans.csv")

    delta = 1.5*abs(data.max()-data.min())

    print(delta)

    # np.random.shuffle(data)

    stream = [[i, {'att': float(line)}] for i, line in enumerate(data)]

    doca = Doca(delta_time=1000, beta=50, mi=100, budget=1, sensitivity=delta)
    doca.cluster(stream, output_dir)


if __name__ == '__main__':

    proc("info_loss")
    print("finish all")
