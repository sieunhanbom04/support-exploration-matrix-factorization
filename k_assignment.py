from pyfaust.factparams import *
from pyfaust.proj import proj_gen
import numpy as np
import queue

# ---------------------------------------- Generalized Hungarian Method ----------------------------------------------------
# This is the code for Algorithm 2 in the paper "Structured Support Exploration For Multilayer Sparse Matrix Factorization". 
# Parameters:
# shape (N,N) = shape of a square matrix
# K = number of non-zero coefficients per row and column 
# C = the matrix we would like to project onto the set of k-sparse matrices (i.e having at most $k$-nonzero coefficient)
# __call__() returns a binary matrix (N,N) indicating the positions where the coefficients are unchanged (and the remaining 
# coefficients are set to zero).

EPS = 1e-7 #Tolerance
INF = 1e10

class Kcolrow(proj_gen):
    def __init__(self, shape, K):
        assert len(shape) == 2
        assert shape[0] == shape[1]
        assert K > 0
        self.constraint = ConstraintInt('splincol', *shape, K)
        self.K = K
        self.shape = shape
        self.size = self.shape[0]

    def __call__(self, C):
        matching = self.assign(-(C * C.conj()).real)
        result = matching * C

        return result

    def assign(self, C):
        #Initialize the matching array (initially zeros), size n x n
        #matching[i][j] = 1 iff (i,j) is chosen
        matching = np.zeros(self.shape).astype(int)

        #potential: size 2 x n
        #potential[0]: potential of row vertices (in a bipartite graph)
        #potential[1]: potential of column vertices (in a bipartite graph)
        potential = np.zeros((2, self.size))
        potential[0] = np.min(C)


        #degree: number of edges in the matching of vertices, size 2 x n
        #degree[0]: degree of row vertices (in a bipartite graph)
        #degree[1]: degree of column vertices (in a bipartite graph)
        degree = np.zeros((2, self.size)).astype(int)

        for _ in range(self.K * self.size):
            q = queue.SimpleQueue()
            parent = np.full((2, self.size), -1).astype(int)
            visited = np.full((2, self.size), False)
            slack = np.full((2, self.size), INF)
            unmatched_vertex = -1

            for v in range(self.size):
                if degree[0, v] < self.K:
                    q.put((0, v))
                    visited[0, v] = True
            
            # perform BFS to find reachable vertices from unsaturated vertices
            while(True):
                # if there is not any reachable vertices, change the potential
                if q.empty():
                    delta = np.inf
                    for side in range(2):
                        for vertex in range(self.size):
                            if not visited[side, vertex]:
                                delta = min(delta, slack[side, vertex])
                    slack = slack - delta
                    for vertex in range(self.size):
                        if visited[0, vertex]:
                            potential[0, vertex] += delta
                        if visited[1, vertex]:
                            potential[1, vertex] -= delta

                    for side in range(2):
                        for vertex in range(self.size):
                            if abs(slack[side, vertex]) < EPS and (not visited[side, vertex]):
                                visited[side, vertex] = True
                                q.put((side, vertex))

                side, vertex = q.get()
                if side == 1 and degree[side, vertex] < self.K:
                    # when we found an augmenting path
                    unmatched_vertex = vertex
                    degree[1, unmatched_vertex] += 1
                    break
                if side == 0:
                    weight = C[vertex]
                    connected = matching[vertex]
                else:
                    weight = C[:, vertex]
                    connected = 1 - matching[:, vertex]

                for u in range(self.size):
                    p_diff = weight[u] - potential[side,vertex] - potential[1 - side, u]
                    if abs(p_diff) > EPS:
                        if side == 0 and (not visited[1 - side, u]) and p_diff > 0 and p_diff < slack[1 -side, u]:
                            slack[1, u] = p_diff
                            parent[1, u] = vertex
                        if side == 1 and (not visited[1 - side, u]) and p_diff < 0 and -p_diff < slack[1-side,u]:
                            slack[0, u] = -p_diff
                            parent[0, u] = vertex
                        continue

                    if (visited[1 - side, u] or connected[u] == 1):
                        continue
                    
                    q.put((1 - side, u))
                    parent[1 - side, u] = vertex
                    visited[1 - side, u] = True

            v = unmatched_vertex
            L = np.zeros(C.shape)
            
            # performing augmentation
            while(True):
                u = parent[1, v]
                p = parent[0, u]
                matching[u, v] = 1
                #print(u,v)
                if p == -1:
                    degree[0, u] += 1
                    break
                else:
                    matching[u, p] = 0
                    v = p

        L = np.zeros(C.shape)
        return matching