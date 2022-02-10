# Copyright (c) 2020-2022, INRIA
# All rights reserved.
#
# BSD License 2.0
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of the <copyright holder> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INRIA BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# website: https://faust/inria.fr
# functions author contact: hakim.hadj-djilani@inria.fr
#
# N.B.: this code is provided in FAµST experimental packages (version 3.25.4),
# please use rather the python wrappers pyfaust.fact.palm4msa and pyfaust.fact.hierarchical
# of the C++ implementations as they are maintained contrary to the functions in
# this module.

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix
import pyfaust
from pyfaust import Faust


def hierarchical_py(A, J, N, res_proxs, fac_proxs, is_update_way_R2L=False,
                    is_fact_side_left=False, compute_2norm_on_arrays=False,
                    norm2_max_iter=100, norm2_threshold=1e-6, use_csr=True,
                    dev='cpu'):
    """
    PALM4MSA-hierarchical factorization.

    Args:
        A: (np.ndarray) the matrix to factorize. The dtype can be np.float32 or np.float64 (it might change the performance).
        J: number of factors.
        N: number of iterations of PALM4MSA calls.
        res_proxs: the residual factor proximity operators.
        fac_proxs: the main factor proximity operators.
        is_update_way_R2L: True to update factors from the right to left in PALM4MSA, False for the inverse order.
        is_fact_side_left: True to slit the left-most factor in the hierarchical factorization, False to split right-most factor.
        compute_2norm_on_arrays: True to compute left/right factors of the current one updated as arrays instead of computing the Faust norms.
        norm2_max_iter: the maximum number of iterations of the 2-norm computation algorithm.
        norm2_threshold: the threshold stopping criterion of the 2-norm computation algorithm.
        use_csr: True to update factors as CSR matrix in PALM4MSA.
        dev: 'cpu' or 'gpu', to use CPU or GPU Faust in the algorithm.

    Returns:
        the Faust resulting from the factorization of A.
    """
    S = Faust([A], dev=dev, dtype=A.dtype)
    l2_ = 1
    compute_2norm_on_arrays_ = compute_2norm_on_arrays
    for i in range(J-1):
        if(isinstance(compute_2norm_on_arrays, list)):
            compute_2norm_on_arrays_ = compute_2norm_on_arrays[i]
        print("hierarchical_py factor", i+1)
        if(is_fact_side_left):
            Si = S.factors(0)
            split_proxs = [res_proxs[i], fac_proxs[i]]
        else:
            Si = S.factors(i)
            split_proxs = [fac_proxs[i], res_proxs[i]]
        Si, l_ = palm4msa_py(Si, 2, N, split_proxs, is_update_way_R2L,
                             S='zero_and_ids', _lambda=1,
                             compute_2norm_on_arrays=compute_2norm_on_arrays_,
                             norm2_max_iter=norm2_max_iter,
                             norm2_threshold=norm2_threshold, use_csr=use_csr,
                             dev=dev)
        l2_ *= l_
        if i > 1:
            if is_fact_side_left:
                S = Si@S.right(1)
            else:
                S = S.left(i-1)@Si
        elif i > 0:
            if is_fact_side_left:
                S = Si@Faust(S.right(1), dev=dev)
            else:
                S = Faust(S.left(i-1), dev=dev)@Si
        else:  # i == 0
            S = Si
        S = S*1/l_
        if is_fact_side_left:
            fp = [*fac_proxs[0:i+1]]
            fp = list(reversed(fp))
            n_proxs = [res_proxs[i], *fp]
        else:
            n_proxs = [*fac_proxs[0:i+1],
                       res_proxs[i]]
            S, l2_ = palm4msa_py(A, S.numfactors(), N, n_proxs,
                                 is_update_way_R2L, S=S, _lambda=l2_,
                                 compute_2norm_on_arrays=compute_2norm_on_arrays_,
                                 norm2_max_iter=norm2_max_iter,
                                 norm2_threshold=norm2_threshold, use_csr=use_csr,
                                 dev=dev)
        S = S*1/l2_
    S = S*l2_
    return S


def palm4msa_py(A, J, N, proxs, is_update_way_R2L=False, S=None, _lambda=1,
                compute_2norm_on_arrays=False, norm2_max_iter=100,
                norm2_threshold=1e-6, use_csr=True, dev='cpu'):
    """
    PALM4MSA factorization.

    Args:
        A: (np.ndarray) the matrix to factorize. The dtype can be np.float32 or np.float64 (it might change the performance).
        J: number of factors.
        N: number of iterations of PALM4MSA.
        proxs: the factor proximity operators.
        is_update_way_R2L: True to update factors from the right to left in PALM4MSA, False for the inverse order.
        S: The Faust (sequence of factors) to initialize the PALM4MSA. By
        default, the first factor to be updated is zero, the other are the
        identity/eye matrix.
        compute_2norm_on_arrays: True to compute left/right factors of the current one updated as arrays instead of computing the Faust norms.
        norm2_max_iter: the maximum number of iterations of the 2-norm computation algorithm.
        norm2_threshold: the threshold stopping criterion of the 2-norm computation algorithm.
        use_csr: True to update factors as CSR matrix in PALM4MSA.
        dev: 'cpu' or 'gpu', to use CPU or GPU Faust in the algorithm.

    Returns:
        the Faust resulting from the factorization of A and its scale factor
        lambda (note that lambda is already applied to the output Faust. It is
        returned only for information, which is useful in hierarchical_py).
    """

    dims = [(prox.constraint._num_rows, prox.constraint._num_cols) for prox in
            proxs]
    A_H = A.T.conj()
    if not isinstance(A_H, np.ndarray):
        A_H = A_H.toarray()
    if S == 'zero_and_ids' or S is None:
        # start Faust, identity factors and one zero
        if is_update_way_R2L:
            S = Faust([np.eye(dims[i][0], dims[i][1], dtype=A.dtype) for i in
                       range(J-1)]+[np.zeros((dims[J-1][0], dims[J-1][1]), dtype=A.dtype)],
                      dev=dev)
        else:
            S = Faust([np.zeros((dims[0][0], dims[0][1]), dtype=A.dtype)] +
                      [np.eye(dims[i+1][0],
                              dims[i+1][1], dtype=A.dtype)
                       for i in
                       range(J-1)],
                      dev=dev)
    lipschitz_multiplicator = 1.001
    for i in range(N):
        if is_update_way_R2L:
            iter_ = reversed(range(J))
        else:
            iter_ = range(J)
        for j in iter_:
            if j == 0:
                S_j = S.factors(j)
                R = S.right(j+1)
                L = np.eye(S_j.shape[0], S_j.shape[0], dtype=A.dtype)
            elif(j == S.numfactors()-1):
                S_j = S.factors(j)
                R = np.eye(S_j.shape[1], S_j.shape[1], dtype=A.dtype)
                L = S.left(j-1)
            else:
                L = S.left(j-1)
                R = S.right(j+1)
                S_j = S.factors(j)
            if not pyfaust.isFaust(L):
                L = Faust(L, dev=dev, dtype=A.dtype)
            if not pyfaust.isFaust(R):
                R = Faust(R, dev=dev, dtype=A.dtype)
            if compute_2norm_on_arrays:
                c = \
                        lipschitz_multiplicator*_lambda**2*norm(R.toarray(), 2)**2 * \
                        norm(L.toarray(), 2)**2

            else:
                c = \
                        lipschitz_multiplicator*_lambda**2*R.norm(2, max_num_its=norm2_max_iter,
                                                                  threshold=norm2_threshold)**2 * \
                        L.norm(2, max_num_its=norm2_max_iter, threshold=norm2_threshold)**2
            if np.isnan(c) or c == 0:
                raise Exception("Failed to compute c (inverse of descent step),"
                                "it could be because of the Faust 2-norm error,"
                                "try option compute_2norm_on_arrays=True")
            if not isinstance(S_j, np.ndarray):
                S_j = S_j.toarray()
            D = S_j-_lambda*(L.H@(_lambda*L@(S_j@R)-A)@R.H)*1/c
            if(not isinstance(D, np.ndarray)):
                D = D.toarray()
            S_j = proxs[j](D)
            if use_csr:
                S_j = csr_matrix(S_j)
            if S.numfactors() > 2 and j > 0 and j < S.numfactors()-1:
                S = L@Faust(S_j, dev=dev, dtype=A.dtype)@R
            elif j == 0:
                S = Faust(S_j, dev=dev, dtype=A.dtype)@R
            else:
                S = L@Faust(S_j, dev=dev, dtype=A.dtype)
        _lambda = np.trace(A_H@S).real/S.norm()**2
    S = _lambda*S
    return S, _lambda
