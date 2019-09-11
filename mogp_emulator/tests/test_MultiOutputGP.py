from tempfile import TemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_allclose
from .. import MultiOutputGP

def test_MultiOutputGP_init():
    "Test function for correct functioning of the init method of MultiOutputGP"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2., 4.]), (2, 1))
    gp = MultiOutputGP(x, y)
    for emulator, yvals in zip(gp.emulators, y):
        assert_allclose(emulator.inputs, x)
        assert_allclose(emulator.targets, yvals)
        assert emulator.nugget == None
    assert gp.n_emulators == 2
    assert gp.n == 1
    assert gp.D == 3
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    gp = MultiOutputGP(x, y)
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3
    assert gp.emulators[0].nugget == None
    
    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([[2., 4.]]),
                                      nugget = np.array([None], dtype = object))
        tmp.seek(0)
        gp = MultiOutputGP(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3
    assert gp.emulators[0].nugget == None
    
    with pytest.raises(AttributeError):
        gp.emulators[0].current_theta

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([[2., 4.]]),
                                      nugget = np.array([1.e-6], dtype = object),
                                      theta = np.array([[1., 2., 3., 4.]]))
        tmp.seek(0)
        gp = MultiOutputGP(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    theta = np.array([1., 2., 3., 4.])
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert_allclose(gp.emulators[0].nugget, 1.e-6)
    assert_allclose(gp.emulators[0].current_theta, theta)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3

def test_MultiOutputGP_init_failures():
    "Tests of MultiOutputGP init method that should fail"

    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    z = np.array([3.])
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y, z, z)
        
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    z = np.array([3., 4.])
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y, z)

    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2., 3., 4.]), (1, 3))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (6,))
    y = np.reshape(np.array([2., 3., 4., 5.]), (2, 2))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3.]), (2, 1))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3., 4., 5.]), (4,))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)

def test_MultiOutputGP_save_emulators():
    "Test function for the save_emulators method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([[2., 4.]])
    gp = MultiOutputGP(x, y)
    
    with TemporaryFile() as tmp:
        gp.save_emulators(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        assert emulator_file['nugget'][0] == None
        with pytest.raises(KeyError):
            emulator_file['theta']
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([np.array([-15.258941170727503, -98.2891773079696  , -56.75514771203786 ,
        13.350449073864349 ])])
    gp._set_params(theta)
    gp.emulators[0].set_nugget(1.e-6)
    
    with TemporaryFile() as tmp:
        gp.save_emulators(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        assert_allclose(np.array(emulator_file['nugget'], dtype=float), 1.e-6)
        assert_allclose(emulator_file['theta'], theta)

def test_MultiOutputGP_get_n_emulators():
    "Test function for the get_n_emulators method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 1
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 4., 6., 8.]), (2, 2))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 2
    
def test_MultiOutputGP_get_n():
    "Test function for the get_n method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n() == 1
    
def test_MultiOutputGP_get_D():
    "Test function for the get_D method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_D() == 3

def test_MultiOutputGP_get_nugget():
    "Test function for the get_nugget method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_nugget() == [None]
    
    gp.emulators[0].set_nugget(1.e-6)
    assert_allclose(np.array(gp.get_nugget()), 1.e-6)
    
def test_MultiOutputGP_set_nugget():
    "Test function for the set_nugget method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    gp.set_nugget([None])
    assert gp.emulators[0].get_nugget() == None
    
    gp.set_nugget([1.e-6])
    assert_allclose(np.array(gp.emulators[0].get_nugget()), 1.e-6)

def test_MultiOutputGP_set_params():
    "Test function for the _set_params method"
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([[2., 3., 4.]])
    gp = MultiOutputGP(x, y)
    theta = np.zeros(4)
    loglike_expected = [17.00784177045409]
    theta_expected = np.zeros(4)
    gp._set_params(theta)
    for emulator, loglike_exp, theta_exp in zip(gp.emulators, loglike_expected, theta_expected):
        assert_allclose(emulator.current_loglikelihood, loglike_exp, atol = 1.e-8, rtol = 1.e-5)
        assert_allclose(emulator.current_theta, theta_exp, atol = 1.e-8, rtol = 1.e-5)
        
def test_MultiOutputGP_set_params_failures():
    "Test function for the _set_params method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([np.array([-15.25894117, -98.28917731, -56.75514771,  13.35044907, -21.60315425])])
    with pytest.raises(AssertionError):
        gp._set_params(theta)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([[-15.25894117, -98.28917731, -56.75514771,  13.35044907],
                      [-15.25894117, -98.28917731, -56.75514771,  13.35044907]])
    with pytest.raises(AssertionError):
        gp._set_params(theta)

def test_MultiOutputGP_learn_hyperparameters():
    "Test function for the learn_hyperparameters method"
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([[2., 3., 4.]])
    gp = MultiOutputGP(x, y)
    theta = np.zeros(4)
    l = gp.learn_hyperparameters(n_tries = 1, theta0 = theta, processes = 1)
    loglike, theta = [list(t) for t in zip(*l)]
    loglike_expected = [4.457233158665504]
    theta_expected = [np.array([ -2.770116256891518, -23.448555866578715, -26.827585590412895, 2.035943563707568])]
    for emulator, loglike_val, theta_val, loglike_exp, theta_exp in zip(gp.emulators, loglike, theta, loglike_expected, theta_expected):
        assert_allclose(loglike_val, loglike_exp, atol = 1.e-8, rtol = 1.e-5)
        assert_allclose(theta_val, theta_exp, atol = 1.e-8, rtol = 1.e-5)
        assert_allclose(emulator.current_loglikelihood, loglike_exp, atol = 1.e-8, rtol = 1.e-5)
        assert_allclose(emulator.current_theta, theta_exp, atol = 1.e-8, rtol = 1.e-5)
        
def test_MultiOutputGP_learn_hyperparameters_failures():
    "Test function for the learn_hyperparameters method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(n_tries = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(processes = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(theta0 = [1., 2.])

def test_MultiOutputGP_predict():
    "Test function for the predict method"
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([[2., 3., 4.]])
    theta = [np.zeros(4)]
    x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
    predict_expected = np.array([[1.395386477054048, 1.7311400058360489]])
    var_expected = np.array([[0.816675395381421, 0.8583559202639046]])
    deriv_expected = np.array([[[6.66666432e-01, 5.81212843e-37, 6.34358943e-19]]])
    gp = MultiOutputGP(x, y)
    gp._set_params(theta)
    predict_actual, var_actual, deriv_actual = gp.predict(x_star)
    
    delta = 1.e-8
    predict_1, _, _ = gp.predict(np.array([[1.-delta, 3., 2.], [3.-delta, 2., 1.]]), do_deriv=False, do_unc=False)
    predict_2, _, _ = gp.predict(np.array([[1., 3.-delta, 2.], [3., 2.-delta, 1.]]), do_deriv=False, do_unc=False)
    predict_3, _, _ = gp.predict(np.array([[1., 3., 2.-delta], [3., 2., 1.-delta]]), do_deriv=False, do_unc=False)
    
    deriv_fd = np.reshape(np.transpose(np.array([(predict_actual - predict_1)/delta, (predict_actual - predict_2)/delta,
                         (predict_actual - predict_3)/delta])), (1, 2, 3))
    
    assert_allclose(predict_actual, predict_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(var_actual, var_expected, atol = 1.e-8, rtol = 1.e-5)
    assert_allclose(deriv_actual, deriv_fd, atol=1.e-8, rtol=1.e-5)
    
    predict_actual, var_actual, deriv_actual = gp.predict(x_star, do_deriv = False, do_unc = False)
    
    assert_allclose(predict_actual, predict_expected, atol = 1.e-8, rtol = 1.e-5)
    assert var_actual is None
    assert deriv_actual is None

def test_MultiOutputGP_predict_failures():
    "Test function for the predict method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3)) 
    x_star = np.reshape(np.array([2., 3.]), (2,))
    theta = [np.zeros(4)]
    gp = MultiOutputGP(x, y)
    gp._set_params(theta)
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star)
    
    x_star = np.reshape(np.array([2., 3., 4., 6., 7., 8.]), (3, 2))
    gp = MultiOutputGP(x, y)
    theta = [np.zeros(4)]
    gp._set_params(theta)
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star)
        
    x_star = np.reshape(np.array([2., 3., 4.]), (1, 3))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters()
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star, processes = -1)

def test_MultiOutputGP_str():
    "Test function for string method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert (str(gp) == "Multi-Output Gaussian Process with:\n1 emulators\n1 training examples\n3 input variables")