import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from .pivot_lapack import lapack_pivot_cholesky

def check_cholesky_inputs(A):
    "Check inputs to cholesky routines"
    
    A = np.array(A)
    assert A.ndim == 2, "A must have shape (n,n)"
    assert A.shape[0] == A.shape[1], "A must have shape (n,n)"
    
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        
    return A

def jit_cholesky(Q, maxtries = 5):
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
    
    :param Q: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type Q: ndarray
    :param maxtries: (optional) Maximum allowable number of attempts to stabilize the Cholesky
                     Decomposition. Must be a positive integer (default = 5)
    :type maxtries: int
    :returns: Lower-triangular factored matrix (shape ``(n,n)`` and the noise that was added to
              the diagonal to achieve that result.
    :rtype: tuple containing an ndarray and a float
    """
    
    Q = check_cholesky_inputs(Q)
    assert int(maxtries) > 0, "maxtries must be a positive integer"
    
    Q = np.ascontiguousarray(Q)
    L, info = lapack.dpotrf(Q, lower = 1)
    if info == 0:
        return L, 0.
    else:
        diagQ = np.diag(Q)
        jitter = diagQ.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(Q + np.eye(Q.shape[0]) * jitter, lower=True)
                return L, jitter
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")

    return L, jitter
    
def pivot_cholesky(A):
    "Pivoted cholesky decomposition routine"
    
    A = check_cholesky_inputs(A)
    
    A = np.ascontiguousarray(A)
    
    L = np.copy(A)
    P = np.empty(A.shape[0], dtype = np.intc)
    
    status = lapack_pivot_cholesky(L, P)
    
    if status < 0:
        raise LinAlgError("Illegal value found in pivoted Cholesky decomposition")
    
    return L, P