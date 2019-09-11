import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..linalg.cholesky import jit_cholesky
from scipy import linalg

def test_jit_cholesky():
    "Tests the stabilized Cholesky decomposition routine"
    
    L_expected = np.array([[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]])
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 0.)
    
    L_expected = np.array([[1.0000004999998751e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                         [9.9999950000037496e-01, 1.4142132088085626e-03, 0.0000000000000000e+00],
                         [6.7379436301144941e-03, 4.7644444411381860e-06, 9.9997779980004420e-01]])
    input_matrix = np.array([[1.                , 1.                , 0.0067379469990855],
                             [1.                , 1.                , 0.0067379469990855],
                             [0.0067379469990855, 0.0067379469990855, 1.                ]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 1.e-6)
    
    input_matrix = np.array([[1.e-6, 1., 0.], [1., 1., 1.], [0., 1., 1.e-10]])
    with pytest.raises(linalg.LinAlgError):
        jit_cholesky(input_matrix)
        
    input_matrix = np.array([[-1., 2., 2.], [2., 3., 2.], [2., 2., -3.]])
    with pytest.raises(linalg.LinAlgError):
        jit_cholesky(input_matrix)

