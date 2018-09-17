import doca
import os
from multiprocessing import Pool
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

    data = np.loadtxt(fname="data/backblaze_2017_q1_m1.csv")

    delta = 1.5*abs(data.max()-data.min())

    print(delta)

    np.random.shuffle(data)

    stream = [[i, {'att': float(line)}] for i, line in enumerate(data)]

    doca.cluster(stream, delta=1000, beta=50, mi=100, eps=1,
                 bounded_delta=delta, output_path=output_dir)


if __name__ == '__main__':
    pool = Pool(3)
    pool.map(proc, [0, 1, 2, 3, 4])

    print("finish all")
