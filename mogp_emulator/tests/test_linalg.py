import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..linalg.cholesky import jit_cholesky, check_cholesky_inputs, pivot_cholesky
from ..linalg.pivot_lapack import lapack_pivot_cholesky
from scipy import linalg

def test_check_cholesky_inputs():
    "Test function that checks inputs to cholesky decomposition routines"
    
    A = np.array([[2., 1.], [1., 2.]])
    B = check_cholesky_inputs(A)
    
    assert_allclose(A, B)
    
    A = np.array([1., 2.])
    with pytest.raises(AssertionError):
        check_cholesky_inputs(A)
        
    A = np.array([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(AssertionError):
        check_cholesky_inputs(A)
    
    input_matrix = np.array([[-1., 2., 2.], [2., 3., 2.], [2., 2., -3.]])
    with pytest.raises(linalg.LinAlgError):
        check_cholesky_inputs(input_matrix)

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

def test_lapack_pivot_cholesky():
    "Test lower level pivoted cholesky routine that interfaces with lapack"
    
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    L_expected = np.array([[ 9.899494936611665 ,  0.                ,  0.                ],
                           [-4.3436559415745055,  4.258245303082538 ,  0.                ],
                           [-1.616244071283537 ,  1.1693999481734827,  0.1423336335961131]])
    Piv_expected = np.array([3, 2, 1], dtype = np.intc)
    
    L_actual = np.copy(input_matrix)
    Piv_actual = np.empty(3, dtype = np.intc)
    
    info = lapack_pivot_cholesky(L_actual, Piv_actual)
    
    assert info == 0
    assert_allclose(L_actual, L_expected)
    assert np.all(Piv_expected == Piv_actual)
    
    input_matrix = np.array([[1., 1., 1.e-6], [1., 1., 1.e-6], [1.e-6, 1.e-6, 1.]])
    L_expected = np.array([[1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                           [9.9999999999999995e-07, 9.9999999999949996e-01, 0.0000000000000000e+00],
                           [1.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333316667e-01]])
    Piv_expected = np.array([1, 3, 2], dtype=np.intc)    
    L_actual = np.copy(input_matrix)
    Piv_actual = np.empty(3, dtype = np.intc)            
    
    info = lapack_pivot_cholesky(L_actual, Piv_actual)
    
    assert info == 1
    assert_allclose(L_actual, L_expected)
    assert np.all(Piv_expected == Piv_actual)

def test_pivot_cholesky():
    "Tests the higher level pivoted cholesky decomposition routine"
    
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[ 9.899494936611665 ,  0.                ,  0.                ],
                           [-4.3436559415745055,  4.258245303082538 ,  0.                ],
                           [-1.616244071283537 ,  1.1693999481734827,  0.1423336335961131]])
    Piv_expected = np.array([3, 2, 1], dtype = np.intc)
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.all(Piv_expected == Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)
    
    input_matrix = np.array([[1., 1., 1.e-6], [1., 1., 1.e-6], [1.e-6, 1.e-6, 1.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                           [9.9999999999999995e-07, 9.9999999999949996e-01, 0.0000000000000000e+00],
                           [1.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333316667e-01]])
    Piv_expected = np.array([1, 3, 2], dtype=np.intc)               
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.all(Piv_expected == Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)