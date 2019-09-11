import numpy as np
from .Kernel import SquaredExponential
from .linalg.cholesky import jit_cholesky, pivot_cholesky, create_pivot_matrix
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy import linalg

class GaussianProcess(object):
    """
    Implementation of a Gaussian Process Emulator.
    
    This class provides an interface to fit a Gaussian Process Emulator to a set of training
    data. The class can be initialized from either a pair of inputs/targets arrays, or a file
    holding data saved from a previous emulator instance (saved via the ``save_emulator``
    method). Once the emulator has been created, the class provides methods for fitting
    optimal hyperparameters, changing hyperparameter values, making predictions, and other
    calculations associated with fitting and making predictions.
    
    The internal emulator structure involves arrays for the inputs, targets, and hyperparameters.
    Other useful information are the number of training examples ``n`` and the number of input
    parameters ``D``. These parameters are available externally through the ``get_n`` and
    ``get_D`` methods
    
    Example: ::
    
        >>> import numpy as np
        >>> from mogp_emulator import GaussianProcess
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([4., 6.])
        >>> gp = GaussianProcess(x, y)
        >>> print(gp)
        Gaussian Process with 2 training examples and 3 input variables
        >>> gp.get_n()
        2
        >>> gp.get_D()
        3
        >>> np.random.seed(47)
        >>> mogp.learn_hyperparameters()
        (5.140462159403397, array([-13.02460687,  -4.02939647, -39.2203646 ,   3.25809653]))
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> gp.predict(x_predict)
        (array([4.74687618, 6.84934016]), array([0.01639298, 1.05374973]),
        array([[8.91363045e-05, 7.18827798e-01, 3.74439445e-16],
               [4.64005897e-06, 3.74191346e-02, 1.94917337e-17]]))
        
    """
    
    def __init__(self, *args):
        """
        Create a new GP Emulator
        
        Creates a new GP Emulator from either the input data and targets to be fit or a
        file holding the input/targets and (optionally) learned parameter values.
        
        Arguments passed to the ``__init__`` method must be either two arguments which
        are numpy arrays ``inputs`` and ``targets``, described below, three arguments
        which are the same ``inputs`` and ``targets`` arrays plus a float representing
        the ``nugget`` parameter, or a single argument which is the filename (string or file
        handle) of a previously saved emulator.
        
        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation.
        
        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This must be a 1D array of length ``n``.
        
        ``nugget`` is the additional noise added to the emulator targets when fitting. This
        can take on values ``None`` (in which case, noise will be added adaptively to
        stabilize fitting), or a float. If the float is non-negative, a fixed noise level
        will be used. If the nugget is a negative float, the code will attempt a pivoted
        Cholesky decomposition. However, using the pivoted Cholesky decomposition requires
        that you build the ``pivot_lapack`` extension using Cython and have a LAPACK library
        version 3.2 or later installed on your system. This is because the pivoted Cholesky
        routine is not included with Scipy by default. If you attempt to use the pivoted
        Cholesky routine and the extension fails to run, the code will fall back onto the
        adaptive nugget method. If no value is specified for the ``nugget`` parameter,
        ``None`` is the default.
        
        If two or three input arguments ``inputs``, ``targets``, and optionally ``nugget`` are
        given:
        
        :param inputs: Numpy array holding emulator input parameters. Must be 2D with shape
                       ``n`` by ``D``, where ``n`` is the number of training examples and
                       ``D`` is the number of input parameters for each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 1D with length ``n``
        :type targets: ndarray
        :param nugget: Noise to be added to the diagonal or ``None``. A non-negative float
                       specifies the noise level explicitly, a negative float indicates that
                       the pivoted Cholesky routine will be used (if available, if not defaults
                       to adaptive nugget), and if ``None`` is given, the noise will set 
                       adaptively to be as small as possible to ensure stable inversion of the
                       covariance matrix. Optional, default is ``None``.
        
        If one input argument ``emulator_file`` is given:
        
        :param emulator_file: Filename or file object for saved emulator parameters (using
                              the ``save_emulator`` method)
        
        :type emulator_file: str or file
        :returns: New ``GaussianProcess`` instance
        :rtype: GaussianProcess
        
        """
        
        emulator_file = None
        theta = None
        nugget = None
        
        if len(args) == 1:
            emulator_file = args[0]
            inputs, targets, theta, nugget = self._load_emulator(emulator_file)
        elif len(args) == 2 or len(args) == 3:
            inputs = np.array(args[0])
            targets = np.array(args[1])
            if targets.shape == (1,) and len(inputs.shape) == 1:
                inputs = np.reshape(inputs, (1, len(inputs)))
            if not len(inputs.shape) == 2:
                raise ValueError("Inputs must be a 2D array")
            if not len(targets.shape) == 1:
                raise ValueError("Targets must be a 1D array")
            if not len(targets) == inputs.shape[0]:
                raise ValueError("First dimensions of inputs and targets must be the same length")
            if len(args) == 3:
                nugget = args[2]
                if not nugget == None:
                    nugget = float(nugget)
        else:
            raise ValueError("Init method of GaussianProcess requires 1 (file) or 2 (input array, target array) arguments")

        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        
        self.n = self.inputs.shape[0]
        self.D = self.inputs.shape[1]
        
        self.nugget = nugget
        
        self.kernel =  SquaredExponential()
        
        if not (emulator_file is None or theta is None):
            self._set_params(theta)

    def _load_emulator(self, filename):
        """
        Load saved emulator and parameter values from file
        
        Method takes the filename of a saved emulator (using the ``save_emulator`` method).
        The saved emulator may or may not contain the fitted parameters. If there are no
        parameters found in the emulator file, the method returns ``None`` for the
        parameters.
        
        :param filename: File where the emulator parameters are saved. Can be a string
                         filename or a file object.
        :type filename: str or file
        :returns: inputs, targets, and (optionally) fitted parameter values from the
                  saved emulator file
        :rtype: tuple containing 3 ndarrays and a float or 2 ndarrays, a None type, and
                a float (if no theta values are found in the emulator file)
        """
        
        emulator_file = np.load(filename)
        
        try:
            inputs = np.array(emulator_file['inputs'])
            targets = np.array(emulator_file['targets'])
        except KeyError:
            raise KeyError("Emulator file does not contain emulator inputs and targets")
            
        try:
            theta = np.array(emulator_file['theta'])
        except KeyError:
            theta = None
            
        try:
            if emulator_file['nugget'] == None:
                nugget = None
            else:
                nugget = float(emulator_file['nugget'])
        except KeyError:
            nugget = None
            
        return inputs, targets, theta, nugget

    def save_emulator(self, filename):
        """
        Write emulators to disk
        
        Method saves the emulator to disk using the given filename or file handle. The method
        writes the inputs and targets arrays to file. If the model has been assigned parameters,
        either manually or by fitting, those parameters are saved as well. Once saved, the
        emulator can be read by passing the file name or handle to the one-argument ``__init__``
        method.
        
        :param filename: Name of file (or file handle) to which the emulator will be saved.
        :type filename: str or file
        :returns: None
        """
        
        emulator_dict = {}
        emulator_dict['targets'] = self.targets
        emulator_dict['inputs'] = self.inputs
        emulator_dict['nugget'] = self.nugget
        
        try:
            emulator_dict['theta'] = self.theta
        except AttributeError:
            pass
                    
        np.savez(filename, **emulator_dict)

    def get_n(self):
        """
        Returns number of training examples for the emulator
        
        :returns: Number of training examples for the emulator object
        :rtype: int
        """
        
        return self.n
        
    def get_D(self):
        """
        Returns number of inputs for the emulator
        
        :returns: Number of inputs for the emulator object
        :rtype: int
        """
        
        return self.D
        
    def get_params(self):
        """
        Returns emulator parameters
        
        Returns current parameters for the emulator as a numpy array if they have been fit. If no
        parameters have been fit, returns None.
        
        :returns: Current parameter values (numpy array of length ``D + 1``), or ``None`` if the
                  parameters have not been fit.
        :rtype: ndarray or None
        """
        
        try:
            return self.theta
        except AttributeError:
            return None
            
    def get_nugget(self):
        """
        Returns emulator nugget parameter
        
        Returns current value of the nugget parameter. If the nugget is selected adaptively, returns None.
        
        :returns: Current nugget value, either a float or ``None``
        :rtype: float or None
        """
        
        return self.nugget
    
    def set_nugget(self, nugget):
        """
        Set the nugget parameter for the emulator
        
        Method for changing the ``nugget`` parameter for the emulator. When a new emulator is
        initilized, this is set to None.
        
        The ``nugget`` parameter controls how noise is added to the covariance matrix in order to
        stabilize the inversion or smooth the emulator predictions. If ``nugget`` is a non-negative
        float, then that particular value is used for the nugget. Note that setting this parameter
        to be zero enforces that the emulator strictly interpolates between points. Alternatively,
        if ``nugget`` is set to be ``None``, the fitting routine will adaptively make the noise
        parameter as large as is needed to ensure that the emulator can be fit.
        
        :param nugget: Controls how noise is added to the emulator. If ``nugget`` is a nonnegative
                       float, then this manually sets the noise parameter (if negative, this will
                       lead to an error), with ``nugget = 0`` resulting in interpolation with no
                       smoothing noise added. ``nugget = None`` will adaptively select the
                       smallest value of the noise term that still leads to a stable inversion of
                       the matrix. Default behavior is ``nugget = None``.
        :type nugget: None or float
        :returns: None
        :rtype: None
        """
        
        if not nugget == None:
            nugget = float(nugget)
        self.nugget = nugget
    
    def _prepare_likelihood(self):
        """
        Pre-calculates matrices needed for fitting and making predictions
        
        Pre-calculates the matrices needed to compute the log-likelihood and make subsequent
        predictions. This is called any time the hyperparameter values are changed in order
        to ensure that all the information is needed to evaluate the log-likelihood and
        its derivatives, which are needed when fitting the optimal hyperparameters.
        
        The method computes the covariance matrix (assuming a squared exponential kernel) 
        and inverts it using the jittered cholesky decomposition. Some additional information
        is also pre-computed and stored. This method has no inputs and no return value,
        but it does modify the state of the object.
        
        :returns: None
        """
        
        self.Q = self.kernel.kernel_f(self.inputs, self.inputs, self.theta)
        
        P = None
        
        if self.nugget == None:
            L, nugget = jit_cholesky(self.Q)
            self.Z = self.Q + nugget*np.eye(self.n)
        elif self.nugget < 0.:
            try:
                L, Piv = pivot_cholesky(self.Q)
                P = create_pivot_matrix(Piv)
            except (ModuleNotFoundError, AssertionError):
                print("Unable to perform pivoted Cholesky decomposition, defaulting to adaptive nugget")
                L, nugget = jit_cholesky(self.Q)
                self.Z = self.Q + nugget*np.eye(self.n)
                P = None
        else:
            self.Z = self.Q + self.nugget*np.eye(self.n)
            L = linalg.cholesky(self.Z, lower=True)
        
        self.invQ = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        if not P is None:
            self.invQ = np.linalg.multi_dot([P, self.invQ, P.T])
        self.invQt = np.dot(self.invQ, self.targets)
        self.logdetQ = 2.0 * np.sum(np.log(np.diag(L)))
        
    def _set_params(self, theta):
        """
        Method for setting the hyperparameters for the emulator
        
        This method is used to reset the value of the hyperparameters for the emulator and
        update the log-likelihood. It is used after fitting the hyperparameters or when loading
        an emulator from file. Input ``theta`` must be array-like with shape
        ``(D + 1,)``, where ``D`` is the number of input parameters.
        
        :param theta: Parameter values to be used for the emulator. Must be array-like and
                      have shape ``(D + 1,)``
        :type theta: ndarray
        :returns: None
        """
        
        theta = np.array(theta)
        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"
        
        self.theta = theta
        self._prepare_likelihood()
    
    def loglikelihood(self, theta):
        """
        Calculate the negative log-likelihood at a particular value of the hyperparameters
        
        Calculate the negative log-likelihood for the given set of parameters. Calling this
        method sets the parameter values and computes the needed inverse matrices in order
        to evaluate the log-likelihood and its derivatives. In addition to returning the
        log-likelihood value, it stores the current value of the hyperparameters and
        log-likelihood in attributes of the object.
        
        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: negative log-likelihood
        :rtype: float
        """
        
        self._set_params(theta)

        loglikelihood = (0.5 * self.logdetQ +
                         0.5 * np.dot(self.targets, self.invQt) +
                         0.5 * self.n * np.log(2. * np.pi))
        self.current_theta = theta
        self.current_loglikelihood = loglikelihood
        return loglikelihood
    
    def partial_devs(self, theta):
        """
        Calculate the partial derivatives of the negative log-likelihood
        
        Calculate the partial derivatives of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions.
        
        During normal use, the ``partial_devs`` method is called after evaluating the
        ``loglikelihood`` method. The implementation takes advantage of this by storing
        the inverse of the covariance matrix, which is expensive to compute and is used
        by the ``loglikelihood``, ``partial_devs``, and ``hessian`` methods. If the function
        is evaluated with a different set of parameters than was previously used to set
        the log-likelihood, the method calls ``_set_params`` to compute the needed
        information. However, caling ``partial_devs`` does not evaluate the log-likelihood,
        so it does not change the cached values of the parameters or log-likelihood.
        
        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-likelihood (array with shape
                  ``(D + 1,)``)
        :rtype: ndarray
        """
        
        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)
            
        partials = np.zeros(self.D + 1)
        
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)
        
        for d in range(self.D + 1):
            partials[d] = -0.5 * (np.dot(self.invQt, np.dot(dKdtheta[d], self.invQt)) - np.sum(self.invQ * dKdtheta[d]))
        
        return partials
        
    def hessian(self, theta):
        """
        Calculate the Hessian of the negative log-likelihood
        
        Calculate the Hessian of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions. It is also used
        to estimate an appropriate step size when fitting hyperparameters using
        the lognormal approximation or MCMC sampling.
        
        When used in an optimization routine, the ``hessian`` method is called after
        evaluating the ``loglikelihood`` method. The implementation takes advantage of
        this by storing the inverse of the covariance matrix, which is expensive to
        compute and is used by the ``loglikelihood`` and ``partial_devs`` methods as well.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-likelihood, the method calls ``_set_params`` to compute the needed
        information. However, caling ``hessian`` does not evaluate the log-likelihood,
        so it does not change the cached values of the parameters or log-likelihood.
        
        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-likelihood (array with shape
                  ``(D + 1, D + 1)``)
        :rtype: ndarray
        """
        
        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)
            
        hessian = np.zeros((self.D + 1, self.D + 1))
        
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)
        d2Kdtheta2 = self.kernel.kernel_hessian(self.inputs, self.inputs, self.theta)
        
        for d1 in range(self.D + 1):
            for d2 in range(self.D + 1):
                hessian[d1, d2] = 0.5*(np.linalg.multi_dot([self.invQt, 
                                        2.*np.linalg.multi_dot([dKdtheta[d1], self.invQ, dKdtheta[d2]])-d2Kdtheta2[d1, d2],
                                        self.invQt])-
                                        np.trace(np.linalg.multi_dot([self.invQ, dKdtheta[d1], self.invQ, dKdtheta[d2]])
                                                 -np.dot(self.invQ, d2Kdtheta2[d1, d2])))
                
        return hessian
    
    def _learn(self, theta0, method = 'L-BFGS-B', **kwargs):
        """
        Minimize log-likelihood function wrt the hyperparameters
        
        Minimize the negative log-likelihood function, with a starting value given by ``theta0``.
        This is done via any gradient-based method available through scipy (see the scipy
        documentation for details), and options can be passed to the minimization routine.
        The default minimization routine is ``L-BFGS-B'``, but this can be specified.
        
        The method is dumb and returns the last value returned from the minimization routine
        irrespective of any error flags returned by the minimization function. This is not
        necessarily a cause for concern, as (1) the parent routine to this function is
        configured to do a certain number of attempts, taking the best result and (2) application
        of the GP does not require that the hyperparameters are at a true minimum of the
        log-likelihood function, just at a value that leads to predictions that are good enough.
        
        The method returns the hyperparameter values as an array with shape ``(D + 1,)`` 
        and the minimimum negative log-likelihood value found.
        
        :param theta0: Starting value for the minimization routine. Must be an array with shape
                       ``(D + 1,)``
        :type theta0: ndarray
        :param method: Minimization method. Must be a gradient-based method available in the
                       ``scipy.optimize.minimize`` function (optional, default is ``'L-BFGS-B'``
                       with no bounds given)
        :type method: str
        :param **kwargs: Additional keyword arguments to be passed to ``scipy.optimize.minimize``
        :returns: minimum hyperparameter values in an array of shape ``(D + 1,)`` and the
                  minimum negative log-likelihood value
        :rtype: tuple containing a ndarray and a float
        """
        
        self._set_params(theta0)
        
        fmin_dict = minimize(self.loglikelihood, theta0, method = method, jac = self.partial_devs, 
                             options = kwargs)
        
        return fmin_dict['x'], fmin_dict['fun']
    
    def learn_hyperparameters(self, n_tries = 15, theta0 = None, method = 'L-BFGS-B', **kwargs):
        """
        Fit hyperparameters by attempting to minimize the negative log-likelihood
        
        Fits the hyperparameters by attempting to minimize the negative log-likelihood multiple times
        from a given starting location and using a particular minimization method. The best result
        found among all of the attempts is returned, unless all attempts to fit the parameters result
        in an error (see below).
        
        If the method encounters an overflow (this can result because the parameter values stored are
        the logarithm of the actual hyperparameters to enforce positivity) or a linear algebra error
        (occurs when the covariance matrix cannot be inverted, even with the addition of additional
        noise added along the diagonal if adaptive noise was selected by setting the nugget parameter
        to be None), the iteration is skipped. If all attempts to find optimal hyperparameters result
        in an error, then the method raises an exception.
        
        The ``theta0`` parameter is the point at which the first iteration will start. If more than
        one attempt is made, subsequent attempts will use random starting points.
        
        The user can specify the details of the minimization method, using any of the gradient-based
        optimizers available in ``scipy.optimize.minimize``. Any additional parameters beyond the method
        specification can be passed as keyword arguments.
        
        The method returns the minimum negative log-likelihood found and the parameter values at
        which that minimum was obtained. The method also sets the current values of the hyperparameters
        to these optimal values and pre-computes the matrices needed to make predictions.
        
        :param n_tries: Number of attempts to minimize the negative log-likelihood function.
                        Must be a positive integer (optional, default is 15)
        :type n_tries: int
        :param theta0: Initial starting point for the first iteration. If present, must be
                       array-like with shape ``(D + 1,)``. If ``None`` is given, then
                       a random value is chosen. (Default is ``None``)
        :type theta0: None or ndarray
        :param method: Minimization method to be used. Can be any gradient-based optimization
                       method available in ``scipy.optimize.minimize``. (Default is ``'L-BFGS-B'``)
        :type method: str
        :param ``**kwargs``: Additional keyword arguments to be passed to the minimization routine.
                         see available parameters in ``scipy.optimize.minimize`` for details.
        :returns: Minimum negative log-likelihood values and hyperparameters (numpy array with shape
                  ``(D + 1,)``) used to obtain those values. The method also sets the current values
                  of the hyperparameters to these optimal values and pre-computes the matrices needed
                  to make predictions.
        :rtype: tuple containing a float and an ndarray
        
        """
    
        n_tries = int(n_tries)
        assert n_tries > 0, "number of attempts must be positive"
        
        np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')
        
        loglikelihood_values = []
        theta_values = []
        
        theta_startvals = 5.*(np.random.rand(n_tries, self.D + 1) - 0.5)
        if not theta0 is None:
            theta0 = np.array(theta0)
            assert theta0.shape == (self.D + 1,), "theta0 must be a 1D array with length D + 1"
            theta_startvals[0,:] = theta0

        for theta in theta_startvals:
            try:
                min_theta, min_loglikelihood = self._learn(theta, method, **kwargs)
                loglikelihood_values.append(min_loglikelihood)
                theta_values.append(min_theta)
            except linalg.LinAlgError:
                print("Matrix not positive definite, skipping this iteration")
            except FloatingPointError:
                print("Floating point error in optimization routine, skipping this iteration")
                
        if len(loglikelihood_values) == 0:
            raise RuntimeError("Minimization routine failed to return a value")
            
        loglikelihood_values = np.array(loglikelihood_values)
        idx = np.argsort(loglikelihood_values)[0]
        
        self._set_params(theta_values[idx])
        
        return loglikelihood_values[idx], theta_values[idx]
    
    def predict(self, testing, do_deriv = True, do_unc = True):
        """
        Make a prediction for a set of input vectors
        
        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points under consideration and
        ``D`` is the number of inputs to the emulator. If the prediction inputs array has shape
        ``(D,)``, then the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.
        
        Optionally, the emulator can also calculate the uncertainties in the predictions 
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.
        
        As with the fitting, this computation can be done independently for each emulator
        and thus can be done in parallel.
        
        :param testing: Array-like object holding the points where predictions will be made.
                        Must have shape ``(n_predict, D)`` or ``(D,)`` (for a single prediction)
        :type testing: ndarray
        :param do_deriv: (optional) Flag indicating if the derivatives are to be computed.
                         If ``False`` the method returns ``None`` in place of the derivative
                         array. Default value is ``True``.
        :type do_deriv: bool
        :param do_unc: (optional) Flag indicating if the uncertainties are to be computed.
                         If ``False`` the method returns ``None`` in place of the uncertainty
                         array. Default value is ``True``.
        :type do_unc: bool
        :param processes: (optional) Number of processes to use when making the predictions.
                          Must be a positive integer or ``None`` to use the number of
                          processors on the computer (default is ``None``)
        :type processes: int or None
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_predict,)``
                  while the derivatives have shape ``(n_predict, D)``. If the ``do_unc`` or
                  ``do_deriv`` flags are set to ``False``, then those arrays are replaced by
                  ``None``.
        :rtype: tuple
        """
        
        testing = np.array(testing)
        if len(testing.shape) == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert len(testing.shape) == 2
                        
        n_testing, D = np.shape(testing)
        assert D == self.D
        
        exp_theta = np.exp(self.theta)

        Ktest = self.kernel.kernel_f(self.inputs, testing, self.theta)

        mu = np.dot(Ktest.T, self.invQt)
        
        var = None
        if do_unc:
            var = np.maximum(exp_theta[self.D] - np.sum(Ktest * np.dot(self.invQ, Ktest), axis=0), 0.)
        
        deriv = None
        if do_deriv:
            deriv = np.zeros((n_testing, self.D))
            for d in range(self.D):
                aa = (self.inputs[:, d].flatten()[None, :] - testing[:, d].flatten()[:, None])
                c = Ktest * aa.T
                deriv[:, d] = exp_theta[d] * np.dot(c.T, self.invQt)
        return mu, var, deriv
        
    def __str__(self):
        """
        Returns a string representation of the model
        
        :returns: A string representation of the model (indicates number of training examples
                  and inputs)
        :rtype: str
        """
        
        return "Gaussian Process with "+str(self.n)+" training examples and "+str(self.D)+" input variables"
        
