import numpy as np
from scipy import linalg
from scipy.linalg import lapack

def check_cholesky_inputs(A):
    """
    Check inputs to cholesky routines
    
    This function is used by both specialized Cholesky routines to check inputs. It verifies
    that the input is a 2D square matrix and that all diagonal elements are positive (a
    necessary but not sufficient condition for positive definiteness). If these checks pass,
    it returns the input as a numpy array.
    
    :param A: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type A: ndarray or similar
    :returns: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :rtype: ndarray
    """
    
    A = np.array(A)
    assert A.ndim == 2, "A must have shape (n,n)"
    assert A.shape[0] == A.shape[1], "A must have shape (n,n)"
    np.testing.assert_allclose(A.T, A)
    
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        
    return A

def jit_cholesky(A, maxtries = 5):
    """
    Performs Jittered Cholesky Decomposition
    
    Performs a Jittered Cholesky decomposition, adding noise to the diagonal of the matrix as needed
    in order to ensure that the matrix can be inverted. Adapted from code in GPy.
    
    On occasion, the matrix that needs to be inverted in fitting a GP is nearly singular. This arises
    when the training samples are very close to one another, and can be averted by adding a noise term
    to the diagonal of the matrix. This routine performs an exact Cholesky decomposition if it can
    be done, and if it cannot it successively adds noise to the diagonal (starting with 1.e-6 times
    the mean of the diagonal and incrementing by a factor of 10 each time) until the matrix can be
    decomposed or the algorithm reaches ``maxtries`` attempts. The routine returns the lower
    triangular matrix and the amount of noise necessary to stabilize the decomposition.
    
    :param A: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type A: ndarray
    :param maxtries: (optional) Maximum allowable number of attempts to stabilize the Cholesky
                     Decomposition. Must be a positive integer (default = 5)
    :type maxtries: int
    :returns: Lower-triangular factored matrix (shape ``(n,n)`` and the noise that was added to
              the diagonal to achieve that result.
    :rtype: tuple containing an ndarray and a float
    """
    
    A = check_cholesky_inputs(A)
    assert int(maxtries) > 0, "maxtries must be a positive integer"
    
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower = 1)
    if info == 0:
        return L, 0.
    else:
        diagA = np.diag(A)
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L, jitter
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")

    return L, jitter
    
def pivot_cholesky(A):
    r"""
    Pivoted cholesky decomposition routine
    
    Performs a pivoted Cholesky decomposition, where rows and columns are interchanged to
    ensure that the decomposition is stable. Returns the factored matrix (with rows that
    were skipped modified appropriately) and the pivoting array describing the interchanges
    made to the matrix.
    
    On occasion, the matrix that needs to be inverted in fitting a GP is nearly singular. This
    arises when the training samples are very close to one another, and can be averted by
    pivoting to factor the non-collinear part of the matrix and then skipping the remaining values
    that are too similar. This routine performs an exact Cholesky decomposition with row and
    column interchanges if it can be done, and if it cannot it computes the portion of the matrix
    that can be inverted and then modifies the rest to skip the points that could not be factored.
    The routine returns the lower triangular matrix and the pivoting interchanges that were made
    in the calculation.
    
    The pivoting vector is a vector of integers of length ``n``, where each value
    :math:`1 \leq k \leq n` appears exactly once. The matrix ``P`` to perform the interchanges
    is then zero everywhere except ``P[piv[k - 1], k - 1] = 1.`` This vector is returned from the
    routine.
    
    :param A: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type A: ndarray
    :returns: Lower-triangular factored matrix (numpy array of shape ``(n,n)`` and the pivoting
              done to obtain the factorization (vector of integers, see above), such that
              ``P**T * A * P = L**T * L``.
    :rtype: tuple containing an ndarray and a ndarray of integers
    """
    
    from .pivot_lapack import lapack_pivot_cholesky
    
    A = check_cholesky_inputs(A)
    
    A = np.ascontiguousarray(A)
    
    L = np.copy(A)
    P = np.empty(A.shape[0], dtype = np.intc)
    
    status = lapack_pivot_cholesky(L, P)
    
    if status < 0:
        raise LinAlgError("Illegal value found in pivoted Cholesky decomposition")
    
    return L, P
    
def create_pivot_matrix(piv):
    r"""
    Create pivot matrix for pivoted Cholesky decomposition
    
    This function creates the matrix needed to undo the row interchanges that occured
    during a pivoted Cholesky decomposition. The input is a vector of integers of length
    ``n``, where each value :math:`1 \leq k \leq n` appears exactly once. The matrix
    ``P`` to perform the interchanges is then zero everywhere except
    ``P[piv[k - 1], k - 1] = 1.`` This matrix is returned from the routine. If not all
    integers are present, the code raises an error.
    
    :param piv: Pivoting vector returned from LAPACK routine. Must be a 1D vector of
                integers of length ``n``, with each ``1 <= k <= n`` represented
                exactly once.
    :type piv: ndarray of type int or other iterable
    :returns: Pivoting matrix of shape ``(n, n)``, which is zero everywhere except
              `P[piv[k - 1], k - 1] = 1.``
    :rtype: ndarray
    """
    
    piv = np.array(piv, dtype = np.intc)
    assert piv.ndim == 1, "piv must be a 1D vector of integers"
    n = len(piv)
    assert np.all(np.sort(piv) == np.arange(1, n + 1)), "bad values for pivoting vector"
    
    P = np.zeros((n,n)) 
    for i in range(n): 
        P[piv[i] - 1, i] = 1.
    
    return P