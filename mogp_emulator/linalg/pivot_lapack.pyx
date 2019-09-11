# cython: language_level=3

from scipy.linalg.cython_lapack cimport dpstrf
from libc.stdlib cimport malloc, free

cpdef int lapack_pivot_cholesky(double[:, ::1] A, int[:] PIV):
    '''pivoted cholesky factorization of real symmetric positive definite float matrix A

    Parameters
    ----------
    A : memoryview (numpy array)
        n x n matrix to compute cholesky decomposition
        will be modified in place
    PIV : memoryview (numpy array of type intc)
        vector of length n to use within function, will be modified
        in place to hold pivoting information such that P[PIV[k], k] = 1
    '''
    cdef int n = A.shape[0], info, rank
    cdef char uplo = b'U'
    cdef double* work = <double*>malloc(sizeof(double)*2*n)
    cdef double tol = -1.

    try:
        dpstrf(&uplo, &n, &A[0,0], &n, &PIV[0], &rank, &tol, work, &info)
    finally:
        free(work)

    cdef int i, j
    for i in range(n):
        for j in range(n):
            if j >= rank:
                if i == j:
                    A[j, j] = A[j - 1, j - 1]/<double>(j + 1)
                else:
                    A[i, j] = 0.
            if j > i:
                A[i, j] = 0.
    
    return info