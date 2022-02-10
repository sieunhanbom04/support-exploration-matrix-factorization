from pyfaust.fact import hierarchical
from pyfaust.factparams import ParamsHierarchical, StoppingCriterion
from pyfaust.proj import skperm, splincol
from pyfaust import wht
from sys import argv
from numpy import log2
from numpy.linalg import norm
import numpy as np

if __name__ == '__main__':

    if(len(argv) > 1):
        d = int(argv[1])
    else:
        d = 64

    # generate a Hadamard matrix H
    H = wht(d, normed=False).toarray()
    d = H.shape[0]
    n = int(log2(d))
    print(np.linalg.norm(H))
    # set the proximity operators
    fac_projs = []  # for the main factors
    res_projs = []  # for the residual factors

    # skperm is used for all the factors
    for i in range(n-1):
        fac_projs += [skperm((d, d), 2, normalized=True)]
        res_projs += [splincol((d, d), int(d/2**(i+1)), normalized=True)]

    # the number of iterations of PALM4MSA calls
    stop_crit = StoppingCriterion(tol = 1e-11 * np.linalg.norm(H), maxiter = 100)
    stop_crit2 = StoppingCriterion(tol = 1e-11 * np.linalg.norm(H), maxiter = 50)

    # pack all the parmeters
    p = ParamsHierarchical(fac_projs, res_projs, stop_crit,
                           stop_crit2, is_update_way_R2L=True,
                           packing_RL=False, is_verbose = True, factor_format = "dense")
    print(p)
    # start the hierarchial factorization of H
    HF = hierarchical(H, p, backend=2020)

    print("error:", (HF-H).norm()/norm(H))
