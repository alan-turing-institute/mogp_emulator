# cython: language_level=3

from scipy.linalg.cython_lapack cimport dpstrf
from libc.stdlib cimport malloc, free

cpdef int lapack_pivot_cholesky(double[:, ::1] A, int[:] PIV):
    """
    LAPACK interface to pivoted Cholesky factorization
    
    This routine is an interface to the LAPACK routine dpstrf, which computes
    the lower-triangular pivoted Cholesky factorization of a symmetric positive
    definite matrix. (Note that because Python uses C-array ordering while
    LAPACK uses Fortran-array ordering, the ``uplo`` character is the opposite
    of what is expected to reflect this.) This routine provides a basic wrapper
    to the routine that does not provide any checks, and then modifies the
    portion of the matrix (if any) that the routine skips due to collinearity
    of matrix rows. The matrix to be factored is modified in-place, and
    the pivoting vector is also modified in place. The routine also
    returns the integer exit code from the LAPACK routine. The LAPACK code
    also requires work space of the same size as the input matrix which is
    allocated and freed in the function.
    
    Possible exit codes are ``0`` (exited normally), ``1`` (algorithm exited
    before completing entire factorization, this situation is corrected in
    this routine), or ``-1`` (algorithm encountered some illegal value and
    failed). If the exit code is ``1``, the portion of the array that was
    skipped is modified in this function.

    :param A: Numpy array of type double, must be 2D with shape ``(n, n)``.
              The array will be modified in place.
    :type A: ndarray of type ``np.float64``
    :param PIV: Pivoting array, must be 1D with shape ``(n,)`` with type
                ``np.intc``. This array must be allocated but does not
                need to be initialized as it is modified in place.
    :type PIV: ndarray of type ``np.intc``
    :returns: Exit code (integer), see possible values above.
    :rtype: int
    """
    
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