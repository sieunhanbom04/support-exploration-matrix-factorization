from pyfaust.fact import hierarchical, palm4msa
from pyfaust.factparams import ParamsHierarchical, StoppingCriterion, ParamsPalm4MSA
from pyfaust.proj import skperm
from pyfaust import wht
from sys import argv
import numpy as np
from numpy import log2
from numpy.linalg import norm
import time

if __name__ == '__main__':

    if(len(argv) > 1):
        d = int(argv[1])
    else:
        d = 64

    t1 = time.time()
    # generate a Hadamard matrix H
    H = wht(d, normed=False).toarray()
    d = H.shape[0]
    n = int(log2(d))

    # set the proximity operators
    fac_projs = []  # for the main factors

    # skperm is used for all the factors
    for i in range(n):
        fac_projs += [skperm((d, d), 2, normalized=True)]

    # the number of iterations of PALM4MSA calls
    stop_crit = StoppingCriterion(num_its=300)
    # pack all the parmeters
    param = ParamsPalm4MSA(fac_projs, stop_crit, is_update_way_R2L=True, packing_RL=False, is_verbose = True)
    print(param)

    # start the hierarchial factorization of H
    HF = palm4msa(H, param, backend = 2016)

    print("error:", (HF-H).norm()/norm(H))
    t2 = time.time()
    print("Running time: ", t2 - t1)