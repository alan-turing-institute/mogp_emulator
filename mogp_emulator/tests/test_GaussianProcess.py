from tempfile import TemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess
from scipy import linalg

def test_GaussianProcess_init():
    "Test function for correct functioning of the init method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 1
    assert gp.nugget == None
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (3, 2))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 2
    assert gp.n == 3
    assert gp.nugget == None
    
    x = np.array([1., 2., 3., 4., 5., 6.])
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert_allclose(np.array([x]), gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 6
    assert gp.n == 1
    assert gp.nugget == None
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y, 1.)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 1
    assert gp.nugget == 1.
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y, None)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 1
    assert gp.nugget == None
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y, -1.)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 1
    assert_allclose(gp.nugget, -1.)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([2., 4.]),
                                      nugget = None)
        tmp.seek(0)
        gp = GaussianProcess(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    assert_allclose(gp.inputs, x)
    assert_allclose(gp.targets, y)
    assert gp.n == 2
    assert gp.D == 3
    assert gp.nugget == None
    with pytest.raises(AttributeError):
        gp.theta
    
    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([2., 4.]),
                                      theta = np.array([1., 2., 3., 4.]),
                                      nugget = 1.e-6)
        tmp.seek(0)
        gp = GaussianProcess(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    theta = np.array([1., 2., 3., 4.])
    assert_allclose(gp.inputs, x)
    assert_allclose(gp.targets, y)
    assert_allclose(gp.theta, theta)
    assert_allclose(gp.nugget, 1.e-6)
    assert gp.n == 2
    assert gp.D == 3
    

def test_GaussianProcess_init_failures():
    "Tests that GaussianProcess fails correctly with bad inputs"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    z = np.array([4.])
    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, z, z)
        
    x = np.array([1., 2., 3., 4., 5., 6.])
    y = np.array([2., 3.])
    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y)
        
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    y = np.array([2., 3., 4.])
    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y)
        
    x = np.array([[1., 2., 3.]])
    y = np.array([[2., 3.], [4., 5]])
    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y)

def test_GaussianProcess_save_emulators():
    "Test function for the save_emulators method"
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    gp = GaussianProcess(x, y)
    
    with TemporaryFile() as tmp:
        gp.save_emulator(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        assert emulator_file['nugget'] == None
        with pytest.raises(KeyError):
            emulator_file['theta']
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (3,))
    gp = GaussianProcess(x, y, 1.e-6)
    theta = np.zeros(4)
    gp._set_params(theta)
    
    with TemporaryFile() as tmp:
        gp.save_emulator(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        assert_allclose(emulator_file['theta'], theta)
        assert_allclose(emulator_file['nugget'], 1.e-6)

def test_GaussianProcess_get_n():
    "Tests the get_n method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert gp.get_n() == 1

def test_GaussianProcess_get_D():
    "Tests the get_D method of Gaussian Process"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert gp.get_D() == 3

def test_GaussianProcess_get_params():
    "Tests the get_params method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    theta_expected = np.zeros(4)
    gp.theta = np.zeros(4)
    assert_allclose(gp.get_params(), theta_expected)
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert gp.get_params() == None
    
def test_GaussianProcess_get_nugget():
    "Tests the get_nugget method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert gp.get_nugget() == None
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y, 1.e-6)
    assert_allclose(gp.get_nugget(), 1.e-6)
    
def test_GaussianProcess_set_nugget():
    "Tests the set_nugget method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    gp.set_nugget(None)
    assert gp.nugget == None
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    gp.set_nugget(1.e-6)
    assert_allclose(gp.nugget, 1.e-6)

def test_GaussianProcess_prepare_likelihood():
    "Tests the _prepare_likelihood method of Gaussian Process"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    invQ_expected = np.array([[ 1.000167195256076 , -0.0110373516824135, -0.0066164596502281],
                              [-0.0110373516824135,  1.0002452278032625, -0.0110373516824135],
                              [-0.0066164596502281, -0.0110373516824135,  1.0001671952560762]])
    invQt_expected = np.array([1.9407564968639992, 2.934511573315307 , 3.954323806676608 ])
    logdetQ_expected = -0.00029059870020992285
    gp.theta = theta
    gp._prepare_likelihood()
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    gp.set_nugget(0.)
    gp.theta = theta
    gp._prepare_likelihood()
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    gp.set_nugget(-1.)
    gp.theta = theta
    gp._prepare_likelihood()
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    x = np.reshape(np.array([1., 2., 3., 1., 2., 3., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp.theta = theta
    gp._prepare_likelihood()
    invQ_expected = np.array([[ 5.0000025001932273e+05, -4.9999974999687175e+05, -3.3691214036059968e-03],
                              [-4.9999974999687175e+05,  5.0000025001932273e+05, -3.3691214038620732e-03],
                              [-3.3691214036059972e-03, -3.3691214038620732e-03, 1.0000444018785022e+00]])
    invQt_expected = np.array([-4.9999876342845545e+05,  5.0000123658773920e+05, 3.9833320004952104e+00])
    logdetQ_expected = -13.122407278313416
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    gp.set_nugget(1.e-6)
    gp.theta = theta
    gp._prepare_likelihood()
    invQ_expected = np.array([[ 5.0000025001932273e+05, -4.9999974999687175e+05, -3.3691214036059968e-03],
                              [-4.9999974999687175e+05,  5.0000025001932273e+05, -3.3691214038620732e-03],
                              [-3.3691214036059972e-03, -3.3691214038620732e-03, 1.0000444018785022e+00]])
    invQt_expected = np.array([-4.9999876342845545e+05,  5.0000123658773920e+05, 3.9833320004952104e+00])
    logdetQ_expected = -13.122407278313416
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    gp.set_nugget(-1.)
    gp.theta = theta
    gp._prepare_likelihood()
    invQ_expected = np.array([[ 1.0000454019910098e+01, -9.0004086179190885e+00, -6.7382529152945373e-03],
                              [-9.0004086179190885e+00,  9.0004086179190885e+00, 0.0000000000000000e+00],
                              [-6.7382529152945373e-03,  0.0000000000000000e+00, 1.0000454019910097e+00]])
    invQt_expected = np.array([-7.027270825598248 ,  9.000408617919089 ,  3.9867051021334494])
    logdetQ_expected = -2.19731537925696
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    gp.set_nugget(0.)
    gp.theta = theta
    with pytest.raises(linalg.LinAlgError):
        gp._prepare_likelihood()

def test_GaussianProcess_set_params():
    "Tests the _set_params method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    invQ_expected = np.array([[ 1.000167195256076 , -0.0110373516824135, -0.0066164596502281],
                              [-0.0110373516824135,  1.0002452278032625, -0.0110373516824135],
                              [-0.0066164596502281, -0.0110373516824135,  1.0001671952560762]])
    invQt_expected = np.array([1.9407564968639992, 2.934511573315307 , 3.954323806676608 ])
    logdetQ_expected = -0.00029059870020992285
    gp._set_params(theta)
    assert_allclose(theta, gp.theta)
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.ones(4)
    invQ_expected = np.array([[ 3.6787944118074578e-01, -1.7918365128975691e-06, -4.6028125821577508e-07],
                              [-1.7918365128975691e-06,  3.6787944118889743e-01, -1.7918365128975699e-06],
                              [-4.6028125821577513e-07, -1.7918365128975699e-06, 3.6787944118074578e-01]])
    invQt_expected = np.array([0.73575166572692  , 1.1036275725476148, 1.471511468650928 ])
    logdetQ_expected = 2.9999999999509863
    gp._set_params(theta)
    assert_allclose(theta, gp.theta, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    x = np.reshape(np.array([1., 2., 3., 1., 2., 3., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    invQ_expected = np.array([[ 5.0000025001932273e+05, -4.9999974999687175e+05, -3.3691214036059968e-03],
                              [-4.9999974999687175e+05,  5.0000025001932273e+05, -3.3691214038620732e-03],
                              [-3.3691214036059972e-03, -3.3691214038620732e-03, 1.0000444018785022e+00]])
    invQt_expected = np.array([-4.9999876342845545e+05,  5.0000123658773920e+05, 3.9833320004952104e+00])
    logdetQ_expected = -13.122407278313416
    gp._set_params(theta)
    assert_allclose(theta, gp.theta, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    invQ_expected = np.array([[ 5.0000025001932273e+05, -4.9999974999687175e+05, -3.3691214036059968e-03],
                              [-4.9999974999687175e+05,  5.0000025001932273e+05, -3.3691214038620732e-03],
                              [-3.3691214036059972e-03, -3.3691214038620732e-03, 1.0000444018785022e+00]])
    invQt_expected = np.array([-4.9999876342845545e+05,  5.0000123658773920e+05, 3.9833320004952104e+00])
    logdetQ_expected = -13.122407278313416
    gp.set_nugget(1.e-6)
    gp._set_params(theta)
    assert_allclose(gp.invQ, invQ_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.invQt, invQt_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.logdetQ, logdetQ_expected, atol = 1.e-8, rtol = 1.e-5)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(5)
    with pytest.raises(AssertionError):
        gp._set_params(theta)

def test_GaussianProcess_loglikelihood():
    "Test the loglikelihood method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    expected_loglikelihood = 17.00784177045409
    actual_loglikelihood = gp.loglikelihood(theta)
    assert_allclose(actual_loglikelihood, expected_loglikelihood, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.theta, theta, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.current_loglikelihood, expected_loglikelihood, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(gp.current_theta, theta, atol = 1.e-8, rtol = 1.e-5)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(5)
    with pytest.raises(AssertionError):
        gp.loglikelihood(theta)

def test_GaussianProcess_partial_devs():
    "Test the partial derivatives of the loglikelihood method of GaussianProcess"

    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    dtheta = 1.e-6
    partials_fd = np.array([(gp.loglikelihood(theta)-gp.loglikelihood([-dtheta, 0., 0., 0.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([0., -dtheta, 0., 0.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([0., 0., -dtheta, 0.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([0., 0., 0., -dtheta]))/dtheta])
    partials_expected = np.array([0.5226523233529687, 0.38484393200869393, 0.217173572580031, -12.751185721368774])
    partials_actual = gp.partial_devs(theta)
    assert_allclose(partials_actual, partials_expected, rtol = 1.e-5, atol = 1.e-8)
    assert_allclose(partials_actual, partials_fd, rtol = 1.e-5, atol = 1.e-8)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.ones(4)
    gp._set_params(theta)
    dtheta = 1.e-6
    partials_fd = np.array([(gp.loglikelihood(theta)-gp.loglikelihood([1. - dtheta, 1., 1., 1.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([1., 1. - dtheta, 1., 1.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([1., 1., 1. - dtheta, 1.]))/dtheta,
                            (gp.loglikelihood(theta)-gp.loglikelihood([1., 1., 1., 1. - dtheta]))/dtheta])
    partials_expected = np.array([0.00017655025945592195, 0.0001753434945624111, 9.2676341163899e-05, -3.834215961850198])
    partials_actual = gp.partial_devs(theta)
    assert_allclose(partials_actual, partials_expected, rtol = 1.e-5, atol = 1.e-8)
    assert_allclose(partials_actual, partials_fd, rtol = 1.e-5, atol = 1.e-8)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    new_theta = np.ones(4)
    partials_expected = np.array([0.00017655025945592195, 0.0001753434945624111, 9.2676341163899e-05, -3.834215961850198])
    partials_actual = gp.partial_devs(new_theta)
    assert_allclose(partials_actual, partials_expected, rtol = 1.e-5, atol = 1.e-8)

def test_GaussianProcess_hessian():
    "test the hessian method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    dtheta = 1.e-6
    hessian_expected = np.zeros((4, 4))
    hessian_expected[0, 0] = -1.0158836934314412
    hessian_expected[0, 1] = -0.550852654584278
    hessian_expected[0, 2] = -0.2896355405877978
    hessian_expected[0, 3] = -0.5221446503589787
    hessian_expected[1, 0] = -0.550852654584278
    hessian_expected[1, 1] = -0.3605958354413885
    hessian_expected[1, 2] = -0.36835202261668
    hessian_expected[1, 3] = -0.384353092048885
    hessian_expected[2, 0] = -0.2896355405877978
    hessian_expected[2, 1] = -0.36835202261668
    hessian_expected[2, 2] = -0.0713607346004307
    hessian_expected[2, 3] = -0.216844530304092
    hessian_expected[3, 0] = -0.5221446503589788
    hessian_expected[3, 1] = -0.384353092048885
    hessian_expected[3, 2] = -0.21684453030409206
    hessian_expected[3, 3] = 14.251171470190169
    hessian_fd = np.zeros((4, 4))
    hessian_fd[0, 0] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[0])/dtheta
    hessian_fd[0, 1] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[0])/dtheta
    hessian_fd[0, 2] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[0])/dtheta
    hessian_fd[0, 3] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[0])/dtheta
    hessian_fd[1, 0] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[1])/dtheta
    hessian_fd[1, 1] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[1])/dtheta
    hessian_fd[1, 2] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[1])/dtheta
    hessian_fd[1, 3] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[1])/dtheta
    hessian_fd[2, 0] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[2])/dtheta
    hessian_fd[2, 1] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[2])/dtheta
    hessian_fd[2, 2] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[2])/dtheta
    hessian_fd[2, 3] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[2])/dtheta
    hessian_fd[3, 0] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[3])/dtheta
    hessian_fd[3, 1] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[3])/dtheta
    hessian_fd[3, 2] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[3])/dtheta
    hessian_fd[3, 3] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[3])/dtheta
    hessian_actual = gp.hessian(theta)
    assert_allclose(hessian_actual, hessian_expected, rtol = 1.e-5, atol = 1.e-8)
    assert_allclose(hessian_actual, hessian_fd, rtol = 1.e-5, atol = 1.e-8)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = -1.*np.ones(4)
    gp._set_params(theta)
    dtheta = 2.e-5
    hessian_expected = np.zeros((4,4))
    hessian_expected[0, 0] = 2.425105736706968
    hessian_expected[0, 1] = -0.7592369859894426
    hessian_expected[0, 2] = 0.08478040182448324
    hessian_expected[0, 3] = -5.582509570382468
    hessian_expected[1, 0] = -0.7592369859894423
    hessian_expected[1, 1] = 2.1779257865718
    hessian_expected[1, 2] = -0.5537146428134243
    hessian_expected[1, 3] = -3.6647301216097365
    hessian_expected[2, 0] = 0.08478040182448301
    hessian_expected[2, 1] = -0.5537146428134241
    hessian_expected[2, 2] = 1.5652039104733786
    hessian_expected[2, 3] = -1.8439118038868478
    hessian_expected[3, 0] = -5.582509570382468
    hessian_expected[3, 1] = -3.664730121609735
    hessian_expected[3, 2] = -1.8439118038868476
    hessian_expected[3, 3] = 30.208300617396674
    hessian_actual = gp.hessian(theta)
    hessian_fd = np.zeros((4, 4))
    hessian_fd[0, 0] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[0])/dtheta
    hessian_fd[0, 1] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[0])/dtheta
    hessian_fd[0, 2] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[0])/dtheta
    hessian_fd[0, 3] = (gp.partial_devs(theta)[0]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[0])/dtheta
    hessian_fd[1, 0] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[1])/dtheta
    hessian_fd[1, 1] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[1])/dtheta
    hessian_fd[1, 2] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[1])/dtheta
    hessian_fd[1, 3] = (gp.partial_devs(theta)[1]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[1])/dtheta
    hessian_fd[2, 0] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[2])/dtheta
    hessian_fd[2, 1] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[2])/dtheta
    hessian_fd[2, 2] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[2])/dtheta
    hessian_fd[2, 3] = (gp.partial_devs(theta)[2]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[2])/dtheta
    hessian_fd[3, 0] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([dtheta, 0., 0., 0.]))[3])/dtheta
    hessian_fd[3, 1] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., dtheta, 0., 0.]))[3])/dtheta
    hessian_fd[3, 2] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., 0., dtheta, 0.]))[3])/dtheta
    hessian_fd[3, 3] = (gp.partial_devs(theta)[3]-gp.partial_devs(theta-np.array([0., 0., 0., dtheta]))[3])/dtheta
    assert_allclose(hessian_actual, hessian_expected, rtol = 1.e-5, atol = 1.e-8)
    assert_allclose(hessian_actual, hessian_fd, rtol = 2.e-4, atol = 1.e-8)
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    new_theta = np.ones(4)
    hessian_expected = np.zeros((4, 4))
    hessian_expected[0, 0] = -0.0010297819788006634
    hessian_expected[0, 1] = -0.0007149385821463147
    hessian_expected[0, 2] = -0.0002995290148774177
    hessian_expected[0, 3] = -0.0001765500790860304
    hessian_expected[1, 0] = -0.0007149385821463147
    hessian_expected[1, 1] = -0.0007779100486468533
    hessian_expected[1, 2] = -0.0004766253650867711
    hessian_expected[1, 3] = -0.00017534323660931483
    hessian_expected[2, 0] = -0.0002995290148774177
    hessian_expected[2, 1] = -0.00047662536508677103
    hessian_expected[2, 2] = -0.00027159517134287816
    hessian_expected[2, 3] = -9.267617781552426e-05
    hessian_expected[3, 0] = -0.0001765500790860304
    hessian_expected[3, 1] = -0.00017534323660931483
    hessian_expected[3, 2] = -9.267617781552426e-05
    hessian_expected[3, 3] = 5.334215961850198
    hessian_actual = gp.hessian(new_theta)
    assert_allclose(hessian_actual, hessian_expected, rtol = 1.e-5, atol = 1.e-8)

def test_GaussianProcess_learn():
    "Test the _learn method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    min_theta_expected = np.array([ -2.770116256891518, -23.448555866578715, -26.827585590412895, 2.035943563707568])
    min_loglikelihood_expected = 4.457233158665504
    min_theta_actual, min_loglikelihood_actual = gp._learn(theta)
    assert_allclose(min_theta_expected, min_theta_actual, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(min_loglikelihood_expected, min_loglikelihood_actual, atol = 1.e-8, rtol = 1.e-5)

        
def test_GaussianProcess_learn_hyperparameters():
    "Test the learn_hyperparameters method of GaussianProcess"
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    min_theta_expected = np.array([ -2.770116256891518, -23.448555866578715, -26.827585590412895, 2.035943563707568])
    min_loglikelihood_expected = 4.457233158665504
    min_loglikelihood_actual, min_theta_actual = gp.learn_hyperparameters(n_tries = 1, theta0 = theta)
    
    assert_allclose(min_theta_expected, min_theta_actual, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(min_loglikelihood_expected, min_loglikelihood_actual, atol = 1.e-8, rtol = 1.e-5)

    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(5)
    with pytest.raises(AssertionError):
        gp.learn_hyperparameters(theta0 = theta)
        
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    with pytest.raises(AssertionError):
        gp.learn_hyperparameters(n_tries = -1)

def test_GaussianProcess_predict():
    """
    Tests the predict method of GaussianProcess -- note the test only checks the derivatives
    via finite differences rather than analytical results as I have not dug through references
    to find the appropriate expressions
    """
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
    predict_expected = np.array([1.395386477054048, 1.7311400058360489])
    unc_expected = np.array([0.816675395381421, 0.8583559202639046])
    predict_actual, unc_actual, deriv_actual = gp.predict(x_star)
    
    delta = 1.e-8
    predict_1, _, _ = gp.predict(np.array([[1. - delta, 3., 2.], [3. - delta, 2., 1.]]), do_deriv=False, do_unc=False)
    predict_2, _, _ = gp.predict(np.array([[1., 3. - delta, 2.], [3., 2. - delta, 1.]]), do_deriv=False, do_unc=False)
    predict_3, _, _ = gp.predict(np.array([[1., 3., 2. - delta], [3., 2., 1. - delta]]), do_deriv=False, do_unc=False)
    
    deriv_fd = np.transpose(np.array([(predict_actual - predict_1)/delta, (predict_actual - predict_2)/delta,
                         (predict_actual - predict_3)/delta]))
    
    assert_allclose(predict_actual, predict_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(unc_actual, unc_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(deriv_actual, deriv_fd, atol = 1.e-8, rtol = 1.e-5)
    
    predict_actual, unc_actual, deriv_actual = gp.predict(x_star, do_deriv = False, do_unc = False)
    assert_allclose(predict_actual, predict_expected)
    assert unc_actual is None
    assert deriv_actual is None
    
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.ones(4)
    gp._set_params(theta)
    x_star = np.array([4., 0., 2.])
    predict_expected = 0.0174176198731851
    unc_expected = 2.7182302871685224
    predict_actual, unc_actual, deriv_actual = gp.predict(x_star)
    
    delta = 1.e-8
    predict_1, _, _ = gp.predict(np.array([4. - delta, 0., 2.]), do_deriv = False, do_unc = False)
    predict_2, _, _ = gp.predict(np.array([4., 0. - delta, 2.]), do_deriv = False, do_unc = False)
    predict_3, _, _ = gp.predict(np.array([4., 0., 2. - delta]), do_deriv = False, do_unc = False)
    
    deriv_fd = np.transpose(np.array([(predict_actual - predict_1)/delta, (predict_actual - predict_2)/delta,
                                      (predict_actual - predict_3)/delta]))
    
    assert_allclose(predict_actual, predict_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(unc_actual, unc_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(deriv_actual, deriv_fd, atol = 1.e-8, rtol = 1.e-5)

def test_GaussianProcess_predict_failures():
    "Test predict method of GaussianProcess with bad inputs"

    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    x_star = np.array([1., 3., 2., 4.])
    with pytest.raises(AssertionError):
        gp.predict(x_star)
        
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])
    gp = GaussianProcess(x, y)
    theta = np.zeros(4)
    gp._set_params(theta)
    x_star = np.reshape(np.array([1., 3., 2., 4., 5., 7.]), (3, 2))
    with pytest.raises(AssertionError):
        gp.predict(x_star)

def test_GaussianProcess_str():
    "Test function for string method"
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.array([2.])
    gp = GaussianProcess(x, y)
    assert (str(gp) == "Gaussian Process with 1 training examples and 3 input variables")
    
