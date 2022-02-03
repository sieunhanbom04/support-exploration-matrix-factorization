from re import X
import numpy as np
#from pyfaust import *
import time
import matplotlib.pyplot as plt

# projection a matrix into the set of of matrices with at most $k$ nonzero coefficents per column
def projection(A, k):
    result = np.zeros(A.shape)
    order = np.argsort(np.absolute(A), axis = 0)
    row_index = order[-k:].T.flatten()
    col_index = np.tile(np.arange(A.shape[1]).astype(int), (k, 1)).T.flatten()
    result[row_index, col_index] = 1
    return result 

# Hard Thresholding Pursuit to perform support exploration
# A: base matrix
# C: target matrix
# k: controls the sparsity of the factor
# max_iter: maximum number of iteration to do HTP
# use_proj: whether or not to use exact projection to find the best solution given a fixed support
# if use_proj = False, perform grad_iter iterations of gradient descent instead
# step: step for gradient descent in case use_proj = False

def HTP(A, C, k, max_iter = 50, mu = 1e-3, use_proj = False, grad_iter = 10, step = 1e-3):
  x = np.zeros((A.shape[1], C.shape[1]))
  B = np.dot(A.T, A)
  z = np.dot(A.T, C)
  
  for i in range(max_iter):
    temp = x + mu * (z - np.dot(B, x))
    #print(temp)
    alive = projection(temp, k)
    
    new_x = np.zeros(x.shape)
    if use_proj:
      for j in range(x.shape[0]):
        support = np.argsort(alive[:, j])[-k:]
        new_x[support, j] = np.dot(np.linalg.pinv(A[:, support]), C[:, j])
    else:
      new_x = x * 1.0
      for j in range(grad_iter):
        new_x = np.multiply(new_x + step * (z - np.dot(B,new_x)), alive)

    x = new_x
  return x

# Bilinear Hard Thresholding Pursuit for minimizing \|A - XY^T\|_F^2
# such that X and Y has at most k0 and k1 coeffcients per column respectively
# k0: controls the sparsity of X
# k1: controls the sparsity of Y 
# max_iter: maximum number of iterations
# flag = True if performing HTP to explore support
# flag = False if not performing HTP. BHTP will be equivalent to PALM4MSA in this case

def BHTP(M, k0, k1, max_iter = 1000, flag = True, verbose = True):
    B = np.eye(M.shape[1], M.shape[1]) 
    A = np.eye(M.shape[0], M.shape[1])
    
    evol = []
    
    bestA = np.zeros(A.shape)
    bestB = np.zeros(B.shape)
    score = 1.0
    
    for i in range(max_iter):
        
        if i % 50 == 0 and flag:
            # HTP for support exploration of A
            suppA = np.where(A!=0, 1, 0)
            temp = HTP(B.T, M.T, k0, step = 1e-3).T
            suppAnew = np.where(temp!=0, 1, 0)
            # If support does not change, keep the old matrix A
            if i ==0 or np.sum(np.abs(suppA - suppAnew)) > 1e-3:
                A = temp
            
            # HTP for support exploration of B
            suppB = np.where(B!=0, 1, 0)
            temp = HTP(A, M, k1, step = 1e-4)
            suppBnew = np.where(temp!=0, 1, 0)
            # if support does not change, keep the old matrix B
            if i ==0 or np.sum(np.abs(suppB - suppBnew)) > 1e-3:
                B = temp
        
        # balanced the factors
        normA = np.linalg.norm(A)
        normB = np.linalg.norm(B)
        A = A * np.sqrt(normB / normA)
        B = B * np.sqrt(normA / normB)

        # refine the factors using PALM
        for _ in range(20):
            adaptlr = (1 + 1e-3) * np.linalg.norm(B, ord = 2) ** 2
            A = A - (1 / adaptlr) * (A @ B - M) @ B.T
            suppA = projection(A.T,k0).T
            A = A * suppA 
    
            adaptlr = (1 + 1e-3) * np.linalg.norm(A, ord = 2) ** 2
            B = B - (1 / adaptlr) * np.dot(A.T, A @ B - M)
            suppB = projection(B,k1)
            B = B * suppB
        relerror = np.linalg.norm(np.dot(A,B) - M) / np.linalg.norm(M)
        evol.append(relerror)
        if verbose:
            print("Iteration ", i, ": ", np.linalg.norm(np.dot(A,B) - M) / np.linalg.norm(M))
    
    return evol
    
    
