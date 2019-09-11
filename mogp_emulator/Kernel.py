"""
Kernel module, implements a few standard stationary kernels for use with the
``GaussianProcess`` class. At present, kernels can only be selected manually by setting
the ``kernel`` attribute of the GP. The default is to use the ``SquaredExponential``
kernel, but this can be changed once the ``GaussianProcess`` instance is created.
"""

import numpy as np
from scipy.spatial.distance import cdist

class Kernel(object):
    """
    Generic class representing a stationary kernel
    
    This base class implements the necessary scaffolding for defining a stationary kernel.
    Stationary kernels are only dependent on a distance measure between any two points, so 
    the base class holds all the necessary information for doing the distance computation.
    Individual subclasses will implement the functional dependence of the kernel on the
    distance, plus first and second derivatives (if desired) to compute the gradient or
    Hessian of the kernel with respect to the hyperparameters.
    
    This implementation uses a scaled euclidean distance metric. Each individual parameter
    has a hyperparameter scale associated with it that is used in the distance computation.
    If a different metric is to be defined, a new base class needs to be defined that
    implements the ``calc_r``, and optionally ``calc_drdtheta`` and ``calc_d2rdtheta2``
    methods if gradient or Hessian computation is desired. The methods ``kernel_f``,
    ``kernel_gradient``, and ``kernel_hessian`` can then be used to compute the appropriate
    quantities with no further modification.
    
    Note that the Kernel object just collates all of the methods together; the class itself
    does not hold any information on the data point or hyperparamters, which are passed
    directly to the appropriate methods. Thus, no information needs to be provided when
    creating a new ``Kernal`` instance.
    """ 
    def __str__(self):
        """
        Defines a string representation of the kernel
        
        Returns a string representation of the kernel. Note that since the kernel just
        collects methods for kernel evaluations together with no data, this is just a basic
        string that will not change for different instances of the class.
        
        :returns: String representation of the kernel
        :rtype: str
        """
        return "Stationary Kernel"

    def _check_inputs(self, x1, x2, params):
        """
        Common function for checking dimensions of inputs
        
        This function checks the inputs to any kernel evaluation for consistency and ensures
        that all input arrays have the correct dimensionality. It returns the reformatted
        arrays, the number of inputs, and the number of hyperparameters. If the method
        determines that the array dimensions are not all consistent with one another,
        it will raise an ``AssertionError``. This method is called internally whenever
        the kernel is evaluated.
        
        :param x1: First parameter array. Should be a 1-D or 2-D array (1-D is acceptable
                   if either there is only a single point, or each point has only a single
                   parameter). If there is more than one parameter, the last dimension
                   must match the last dimension of ``x2`` and be one less than the length
                   of ``params``.
        :type x1: array-like
        :param x2: Second parameter array. The same restrictions apply that hold for ``x1``
                   described above.
        :type x2: array-like
        :param params: Hyperparameter array. Must have a length that is one more than the
                       last dimension of ``x1`` and ``x2``, hence minimum length is 2.
        :type params: array-like
        :returns: A tuple containing the following: reformatted ``x1``, ``n1``, reformatted
                  ``x2``, ``n2``, ``params``, and ``D``. ``x1`` will be an array with
                  dimensions ``(n1, D - 1)``, ``x2`` will be an array with dimensions
                  ``(n2, D - 1)``, and ``params`` will be an array with dimensions ``(D,)``.
                  ``n1``, ``n2``, and ``D`` will be integers.
        """
        
        params = np.array(params)
        assert params.ndim == 1, "parameters must be a vector"
        D = len(params)
        assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
        x1 = np.array(x1)
    
        assert x1.ndim == 1 or x1.ndim == 2, "bad number of dimensions in input x1"
    
        if x1.ndim == 2:
            assert x1.shape[1] == D - 1, "bad shape for x1"
        else:
            if D == 2:
                x1 = np.reshape(x1, (len(x1), 1))
            else:
                x1 = np.reshape(x1, (1, D - 1))
    
        n1 = x1.shape[0]
            
        x2 = np.array(x2)
    
        assert x2.ndim == 1 or x2.ndim == 2, "bad number of dimensions in input x2"
    
        if x2.ndim == 2:
            assert x2.shape[1] == D - 1, "bad shape for x2"
        else:
            if D == 2:
                x2 = np.reshape(x2, (len(x2), 1))
            else:
                x2 = np.reshape(x2, (1, D - 1))
            
        n2 = x2.shape[0]
        
        return x1, n1, x2, n2, params, D
        

    def calc_r(self, x1, x2, params):
        r"""
        Calculate distance between all pairs of points
        
        This method computes the scaled Euclidean distance between all pairs of points
        in ``x1`` and ``x2``. Each component distance is multiplied by the corresponding
        hyperparameter prior to summing and taking the square root. For example, if
        ``x1 = [1.]``, ``x2`` = [2.], and ``params = [2., 2.]`` then ``calc_r`` would
        return :math:`{\sqrt{2(1 - 2)^2}=\sqrt{2}}` as an array with shape ``(1,1)``.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all pair-wise distances between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        r_matrix = cdist(x1, x2, "seuclidean", V = exp_theta)
    
        return r_matrix

    def calc_drdtheta(self, x1, x2, params):
        """
        Calculate the first derivative of the distance between all pairs of points with
        respect to the hyperparameters
        
        This method computes the derivative of the scaled Euclidean distance between
        all pairs of points in ``x1`` and ``x2`` with respect to the hyperparameters.
        The gradient is held in an array with shape ``(D, n1, n2)``, where ``D`` is
        the length of ``params``, ``n1`` is the length of the first axis of ``x1``,
        and ``n2`` is the length of the first axis of ``x2``. This is used in the
        computation of the gradient and Hessian of the kernel. The first index
        represents the different derivatives with respect to each hyperparameter.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the derivative of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first axis
                  indicates the different derivative components (i.e. the derivative
                  with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        drdtheta = np.zeros((D - 1, n1, n2))
    
        r_matrix = self.calc_r(x1, x2, params)
        r_matrix[(r_matrix == 0.)] = 1.
    
        for d in range(D - 1):
            drdtheta[d] = 0.5 * np.exp(params[d]) / r_matrix * cdist(np.reshape(x1[:,d], (n1, 1)),
                                                                     np.reshape(x2[:,d], (n2, 1)), "sqeuclidean")
                                                                 
        return drdtheta
    
    def calc_d2rdtheta2(self, x1, x2, params):
        """
        Calculate all second derivatives of the distance between all pairs of points with
        respect to the hyperparameters
        
        This method computes all second derivatives of the scaled Euclidean distance
        between all pairs of points in ``x1`` and ``x2`` with respect to the
        hyperparameters. The gradient is held in an array with shape ``(D, D, n1, n2)``,
        where ``D`` is the length of ``params``, ``n1`` is the length of the first axis
        of ``x1``, and ``n2`` is the length of the first axis of ``x2``. This is used in
        the computation of the gradient and Hessian of the kernel. The first two indices
        represents the different derivatives with respect to each hyperparameter.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the second derivatives of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first two axes
                  indicates the different derivative components (i.e. the second derivative
                  with respect to the first parameter is [0,0,:,:], the mixed partial with
                  respect to the first and second parameters is [0,1,:,:] or [1,0,:,:], etc.)
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        d2rdtheta2 = np.zeros((D - 1, D - 1, n1, n2))
    
        r_matrix = self.calc_r(x1, x2, params)
        r_matrix[(r_matrix == 0.)] = 1.
    
        for d1 in range(D - 1):
            for d2 in range(D - 1):
                if d1 == d2:
                    d2rdtheta2[d1, d2] = (0.5*np.exp(params[d1]) / r_matrix *
                                          cdist(np.reshape(x1[:,d1], (n1, 1)),
                                                np.reshape(x2[:,d1], (n2, 1)), "sqeuclidean"))
                d2rdtheta2[d1, d2] -= (0.25 * np.exp(params[d1]) * np.exp(params[d2]) / r_matrix**3 *
                                       cdist(np.reshape(x1[:,d1], (n1, 1)), np.reshape(x2[:,d1], (n2, 1)), "sqeuclidean")*
                                       cdist(np.reshape(x1[:,d2], (n1, 1)), np.reshape(x2[:,d2], (n2, 1)), "sqeuclidean"))
                                                                 
        return d2rdtheta2
        
    def kernel_f(self, x1, x2, params):
        """
        Compute kernel values for a set of inputs
        
        Returns the value of the kernel for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric and then evaluates the kernel function
        for those distances.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all kernel values between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        return np.exp(params[D - 1]) * self.calc_K(self.calc_r(x1, x2, params))
        
    def kernel_deriv(self, x1, x2, params):
        """
        Compute kernel gradient for a set of inputs
        
        Returns the value of the kernel gradient for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the gradient of the kernel function between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is the
                  length of the first axis of ``x1`` and ``n2`` is the length of the first axis
                  of ``x2``. The first axis indicates the different derivative components
                  (i.e. the derivative with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        dKdtheta = np.zeros((D, n1, n2))
    
        dKdtheta[-1] = self.kernel_f(x1, x2, params)
    
        dKdr = self.calc_dKdr(self.calc_r(x1, x2, params))
    
        drdtheta = self.calc_drdtheta(x1, x2, params)
    
        for d in range(D - 1):
            dKdtheta[d] = np.exp(params[-1]) * dKdr * drdtheta[d]
    
        return dKdtheta
        
    def kernel_hessian(self, x1, x2, params):
        """
        Calculate the Hessian of the kernel evaluated for all pairs of points with
        respect to the hyperparameters
        
        Returns the value of the kernel Hessian for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.
        
        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the Hessian of the pair-wise distances between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is
                  the length of the first axis of ``x1`` and ``n2`` is the length of the first
                  axis of ``x2``. The first two axes indicates the different derivative components
                  (i.e. the second derivative with respect to the first parameter is [0,0,:,:],
                  the mixed partial with respect to the first and second parameters is [0,1,:,:]
                  or [1,0,:,:], etc.)
        :rtype: ndarray
        """
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        d2Kdtheta2 = np.zeros((D, D, n1, n2))
    
        d2Kdtheta2[-1, :] = self.kernel_deriv(x1, x2, params)
        d2Kdtheta2[:, -1] = d2Kdtheta2[-1, :]
    
        r_matrix = self.calc_r(x1, x2, params)
        dKdr = self.calc_dKdr(r_matrix)
        d2Kdr2 = self.calc_d2Kdr2(r_matrix)
    
        drdtheta = self.calc_drdtheta(x1, x2, params)
        d2rdtheta2 = self.calc_d2rdtheta2(x1, x2, params)
    
        for d1 in range(D - 1):
            for d2 in range(D - 1):
                d2Kdtheta2[d1, d2] = np.exp(params[-1]) * (d2Kdr2 * drdtheta[d1] * drdtheta[d2] + dKdr * d2rdtheta2[d1, d2])
    
        return d2Kdtheta2

    def calc_K(self, r):
        """
        Calculate kernel as a function of distance
        
        This method implements the kernel function as a function of distance. Given an array
        of distances, this function evaluates the kernel function of those values, returning
        an array of the same shape. Note that this is not implemented for the base class, as
        this must be defined for a specific kernel.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel evaluations, with the same shape as the input ``r``
        :rtype: ndarray
        """
        
        raise NotImplementedError("base Kernel class does not implement a kernel function")
        
    def calc_dKdr(self, r):
        """
        Calculate first derivative of kernel as a function of distance
        
        This method implements the first derivative of the kernel function as a function of
        distance. Given an array of distances, this function evaluates the derivative
        function of those values, returning an array of the same shape. Note that this is
        not implemented for the base class, as this must be defined for a specific kernel.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
        
        raise NotImplementedError("base Kernel class does not implement a kernel derivative function")
        
    def calc_d2Kdr2(self, r):
        """
        Calculate second derivative of kernel as a function of distance
        
        This method implements the second derivative of the kernel function as a function of
        distance. Given an array of distances, this function evaluates the second derivative
        function of those values, returning an array of the same shape. Note that this is
        not implemented for the base class, as this must be defined for a specific kernel.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel second derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
        
        raise NotImplementedError("base Kernel class does not implement kernel derivatives")

class SquaredExponential(Kernel):
    """
    Implementation of the squared exponential kernel
    
    Class representing a squared exponential kernel. It derives from the base class for a
    stationary kernel, using the scaled Euclidean distance metric. The subclass then just
    defines the kernel function and its derivatives.
    """

    def calc_K(self, r):
        """
        Compute K(r) for the squared exponential kernel
        
        This method implements the squared exponential kernel function as a function of distance.
        Given an array of distances, this function evaluates the kernel function of those values,
        returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel evaluations, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return np.exp(-0.5*r**2)
    
    def calc_dKdr(self, r):
        """
        Calculate first derivative of the squared exponential kernel as a function of distance
        
        This method implements the first derivative of the squared exponential kernel function
        as a function of distance. Given an array of distances, this function evaluates the derivative
        function of those values, returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return -r*np.exp(-0.5*r**2)
    
    def calc_d2Kdr2(self, r):
        """
        Calculate second derivative of the squared exponential kernel as a function of distance
        
        This method implements the second derivative of the squared exponential kernel function
        as a function of distance. Given an array of distances, this function evaluates the
        second derivative function of those values, returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel second derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return (r**2 - 1.)*np.exp(-0.5*r**2)
        
    def __str__(self):
        """
        Defines a string representation of the squared exponential kernel
        
        Returns a string representation of the squared exponential kernel. Note that since
        the kernel just collects methods for kernel evaluations together with no data, this
        is just a basic string that will not change for different instances of the class.
        
        :returns: String representation of the kernel
        :rtype: str
        """
        return "Squared Exponential Kernel"

class Matern52(Kernel):
    """
    Implementation of the Matern 5/2 kernel
    
    Class representing the Matern 5/2 kernel. It derives from the base class for a
    stationary kernel, using the scaled Euclidean distance metric. The subclass then just
    defines the kernel function and its derivatives.
    """
    def calc_K(self, r):
        """
        Compute K(r) for the Matern 5/2 kernel
        
        This method implements the Matern 5/2 kernel function as a function of distance.
        Given an array of distances, this function evaluates the kernel function of those values,
        returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel evaluations, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return (1.+np.sqrt(5.)*r+5./3.*r**2)*np.exp(-np.sqrt(5.)*r)
    
    def calc_dKdr(self, r):
        """
        Calculate first derivative of the Matern 5/2 kernel as a function of distance
        
        This method implements the first derivative of the Matern 5/2 kernel function
        as a function of distance. Given an array of distances, this function evaluates the derivative
        function of those values, returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return -5./3.*r*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)
    
    def calc_d2Kdr2(self, r):
        """
        Calculate second derivative of the squared exponential kernel as a function of distance
        
        This method implements the second derivative of the squared exponential kernel function
        as a function of distance. Given an array of distances, this function evaluates the
        second derivative function of those values, returning an array of the same shape.
        
        :param r: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r: array-like
        :returns: Array holding kernel second derivatives, with the same shape as the input ``r``
        :rtype: ndarray
        """
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return 5./3.*(5.*r**2-np.sqrt(5.)*r-1.)*np.exp(-np.sqrt(5.)*r)
        
    def __str__(self):
        """
        Defines a string representation of the Matern 5/2 kernel
        
        Returns a string representation of the Matern 5/2 kernel. Note that since
        the kernel just collects methods for kernel evaluations together with no data, this
        is just a basic string that will not change for different instances of the class.
        
        :returns: String representation of the kernel
        :rtype: str
        """
        return "Matern 5/2 Kernel"

    
