from numpy.linalg import norm
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix
import _FaustCorePy
from pyfaust import *
from pyfaust.factparams import *
from pyfaust.proj import *
from pyfaust.proj import skperm
import matplotlib.pyplot as plt

# this code has been extracted from FAµST experimental package (please do not redistribute)

def hierarchical_py(A, J, N, res_proxs, fac_proxs, is_update_way_R2L=False,
                    is_fact_side_left=False, compute_2norm_on_arrays=False,
                    norm2_max_iter=100, norm2_threshold=1e-6, use_csr=True):
    S = Faust([A])
    l2_ = 1
    compute_2norm_on_arrays_ = compute_2norm_on_arrays
    for i in range(J-1):
        if(isinstance(compute_2norm_on_arrays, list)):
            compute_2norm_on_arrays_ = compute_2norm_on_arrays[i]
        print("hierarchical_py factor", i+1)
        if(is_fact_side_left):
            Si = S.factors(0)
        else:
            Si = S.factors(i)
        Si, l_ = palm4msa_py(Si, 2, N, [fac_proxs[i], res_proxs[i]], is_update_way_R2L,
                    S='zero_and_ids', _lambda=1,
                             compute_2norm_on_arrays=compute_2norm_on_arrays_,
                             norm2_max_iter=norm2_max_iter,
                             norm2_threshold=norm2_threshold, use_csr=use_csr)
        l2_ *= l_
        if i > 1:
            S = S.left(i-1) @ Si
        elif i > 0:
            S = Faust(S.left(i-1)) @ Si
        else: # i == 0
            S = Si
        S = S*1/l_
        S,l2_ = palm4msa_py(A, S.numfactors(), N, [p for p in
                                               [*fac_proxs[0:i+1],
                                                res_proxs[i]]],
                        is_update_way_R2L, S=S, _lambda=l2_,
                            compute_2norm_on_arrays=compute_2norm_on_arrays_,
                             norm2_max_iter=norm2_max_iter,
                             norm2_threshold=norm2_threshold, use_csr=use_csr)
        S = S*1/l2_
    S = S*l2_
    return S

def palm4msa_py(A, J, N, proxs, is_update_way_R2L=False, S=None, _lambda=1,
                compute_2norm_on_arrays=False, norm2_max_iter=100,
                norm2_threshold=1e-6, use_csr=True):
    dims = [(prox.constraint._num_rows, prox.constraint._num_cols) for prox in
            proxs ]
    A_H = A.T.conj()
    if(not isinstance(A_H, np.ndarray)):
       A_H = A_H.toarray()
    if(S == 'zero_and_ids'):
        # start Faust, identity factors and one zero
        if(is_update_way_R2L):
            S = Faust([np.eye(dims[i][0],dims[i][1]) for i in range(J-1)]+[np.zeros((dims[J-1][0], dims[J-1][1]))])
        else:
            S = Faust([np.zeros((dims[0][0],dims[0][1]))]+[np.eye(dims[i+1][0], dims[i+1][1]) for i in range(J-1)])
    elif(S == None):
        # start Faust, identity factors
        S = Faust([np.eye(dims[i][0], dims[i][1]) for i in range(J)])
    lipschitz_multiplicator=1.001
    for i in range(N):
        if(is_update_way_R2L):
            iter_ = reversed(range(J))
        else:
            iter_ = range(J)
        for j in iter_:
            if(j == 0):
                L = np.eye(dims[0][0],dims[0][0])
                S_j = S.factors(j)
                R = S.right(j+1)
            elif(j == S.numfactors()-1):
                L = S.left(j-1)
                S_j = S.factors(j)
                R = np.eye(dims[j][1], dims[j][1])
            else:
                L = S.left(j-1)
                R = S.right(j+1)
                S_j = S.factors(j)
            if(not pyfaust.isFaust(L)): L = Faust(L)
            if(not pyfaust.isFaust(R)): R = Faust(R)
            if(compute_2norm_on_arrays):
                c = \
                        lipschitz_multiplicator*_lambda**2*norm(R.toarray(),2)**2 * \
                        norm(L.toarray(),2)**2

            else:
                c = \
                        lipschitz_multiplicator*_lambda**2*R.norm(2, max_num_its=norm2_max_iter,
                                                                  threshold=norm2_threshold)**2 * \
                        L.norm(2,max_num_its=norm2_max_iter, threshold=norm2_threshold)**2
            if(np.isnan(c) or c == 0):
                raise Exception("Failed to compute c (inverse of descent step),"
                                "it could be because of the Faust 2-norm error,"
                                "try option compute_2norm_on_arrays=True")
            if(not isinstance(S_j, np.ndarray)): # equiv. to use_csr except
                                                 # maybe for the first iteration
                S_j = S_j.toarray()
            D = S_j-_lambda*(L.H @ (_lambda*L @ (S_j @ R)-A) @ R.H)*1/c
            if(not isinstance(D, np.ndarray)):
                D = D.toarray()
            S_j = proxs[j](D)
            if(use_csr):
                S_j = csr_matrix(S_j)
            if(S.numfactors() > 2 and j > 0 and j < S.numfactors()-1):
                S = L @ Faust(S_j) @ R
            elif(j == 0):
                S = Faust(S_j) @ R
            else:
                S = L @ Faust(S_j)
        _lambda = np.trace(A_H*S).real/S.norm()**2

    S = _lambda*S
    return S, _lambda
