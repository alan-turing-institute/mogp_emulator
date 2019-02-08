# Class implementing a multi-output Gaussian Process for emulating the output of
# a series of computer simulations. Since each emulator is independent of the
# others, the emulators can be fit in parallel, significantly speeding up
# the fitting procedure.

# The class relies on an underlying implementation of a gp_emulator which you must
# have installed on your system. I used Jose Gomez-Dans' original implementation,
# but the interface for Sinan's code should be idential. However, I note that Sinan's
# code did not use the most efficient method for forming the covariance matrix, which
# takes up the bulk of the time for fitting each emulator. Therefore, I suggest
# using Jose's code for the time being, as it should give satisfactory performance
# for small problems

from multiprocessing import Pool
import numpy as np
from gp_emulator.GaussianProcess import GaussianProcess

class MultiOutputGP(object):
    """
    Implementation of a multiple-output Gaussian Process Emulator.
    
    This class provides an interface to fit a Gaussian Process Emulator to multiple targets
    using the same input data. The class creates all of the necessary sub-emulators from
    the input data and provides interfaces to the ``learn_hyperparameters`` and ``predict``
    methods of the sub-emulators. Because the emulators are all fit independently, the
    class provides the option to use multiple processes to fit the emulators and make
    predictions in parallel.
    
    Example: ::
    
        >>> import numpy as np
        >>> from mogp_emulator import MultiOutputGP
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([[4., 6.], [5., 7.]])
        >>> mogp = MultiOutputGP(x, y)
        >>> print(mogp)
        Multi-Output Gaussian Process with:
        2 emulators
        2 training examples
        3 input variables
        >>> mogp.get_n_emulators()
        2
        >>> mogp.get_n()
        2
        >>> mogp.get_D()
        3
        >>> np.random.seed(47)
        >>> mogp.learn_hyperparameters()
        After 15, the minimum cost was 5.322784e+00
        After 15, the minimum cost was 5.140462e+00
        [(5.1404621594033895,
          array([-6.95976295, -4.99805894, -5.21415165,  3.23718116, -0.61961335])),
         (5.322783716197344,
          array([-9.80965415, -5.53659105, -6.29521694,  3.58162789,  0.06580016]))]
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> mogp.predict()
        (array([[4.76282574, 6.36038561],
                [5.78892678, 6.9389214 ]]), array([[0.30425374, 2.21021771],
                [0.56260549, 2.67487348]]), array([[[0.03785875, 0.26923002, 0.21690803],
                 [0.00478238, 0.03400961, 0.02740021]],
 
                [[0.00309533, 0.22206213, 0.10399381],
                 [0.00107731, 0.07728762, 0.03619453]]]))
    
    Note that there will frequently be a RuntimeWarning during the fitting of hyperparameters,
    due to the random initial conditions which sometimes leads to a poorly conditioned matrix
    inversion. This should only be a concern if the final minimum cost is 9999 (meaning that
    all attempts to minimize the negative log-likelihood resulted in an error).
        
    """
    def __init__(self, inputs, targets):
        """
        Create a new multi-output GP Emulator
        
        Creates a new multi-output GP Emulator from the input data and targets to be fit.
        
        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation. Because the model assumes all
        outputs are drawn from the same identical set of simulations (i.e. the normal use
        case is to fit a series of computer simulations with multiple outputs from the same
        input), the input to each emulator is identical.
        
        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This can be either a 1D or 2D array, where the last dimension must have length
        ``n``. If the ``targets`` array is of shape ``(n_emulators,n)``, then the emulator fits
        a total of ``n_emulators`` to the different target arrays, while if targets has shape
        ``(n,)``, a single emulator is fit.
        
        :param inputs: Numpy array holding emulator input parameters. Must be 2D with shape
                       ``n`` by ``D``, where ``n`` is the number of training examples and
                       ``D`` is the number of input parameters for each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 2D or 1D with length
                       ``n`` in the final dimension. The first dimension is of length
                       ``n_emulators`` (defaults to a single emulator if the input is 1D)
        """
        
        # check input types and shapes, reshape as appropriate for the case of a single emulator
        inputs = np.array(inputs)
        targets = np.array(targets)
        if len(targets.shape) == 1:
            targets = np.reshape(targets, (1, len(targets)))
        elif not (len(targets.shape) == 2):
            raise ValueError("targets must be either a 1D or 2D array")
        if not (len(inputs.shape) == 2):
            raise ValueError("inputs must be 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        self.emulators = [ GaussianProcess(inputs, single_target) for single_target in targets]
        
        self.n_emulators = targets.shape[0]
        self.n = inputs.shape[0]
        self.D = inputs.shape[1]
        
    def get_n_emulators(self):
        """
        Returns the number of emulators
        
        :returns: Number of emulators in the object
        :rtype: int
        """
        return self.n_emulators
        
    def get_n(self):
        """
        Returns number of training examples in each emulator
        
        :returns: Number of training examples in each emulator in the object
        :rtype: int
        """
        return self.n
        
    def get_D(self):
        """
        Returns number of inputs for each emulator
        
        :returns: Number of inputs for each emulator in the object
        :rtype: int
        """
        return self.D
        
    def learn_hyperparameters(self, n_tries=15, verbose=False, x0=None, processes=None):
        """
        Fit hyperparameters for each model
        """
        
        assert int(n_tries) > 0, "n_tries must be a positive integer"
        if not x0 is None:
            x0 = np.array(x0)
            assert len(x0) == self.D + 2, "x0 must have length of number of input parameters D + 2"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be positive"
        
        n_tries = int(n_tries)
        
        p = Pool(processes)
        likelihood_theta_vals = p.starmap(GaussianProcess.learn_hyperparameters, [(gp, n_tries, verbose, x0) for gp in self.emulators])
        
        # re-evaluate log likelihood for each emulator to update current parameter values
        # (needed because of how multiprocessing works -- the bulk of the work is done in
        # parallel, but this step ensures that the results are gathered correctly for each
        # emulator)
        for emulator, (loglike, theta) in zip(self.emulators, likelihood_theta_vals):
            _ = emulator.loglikelihood(theta)
        return likelihood_theta_vals
        
    def predict(self, testing, do_deriv=True, do_unc=True, processes=None):
        """
        Make a prediction for a set of input vectors
        """
        testing = np.array(testing)
        assert len(testing.shape) == 2, "testing must be a 2D array"
        assert testing.shape[1] == self.D, "second dimension of testing must be the same as the number of input parameters"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"
            
        p = Pool(processes)
        
        predict_vals = p.starmap(GaussianProcess.predict, [(gp, testing) for gp in self.emulators])
        
        # repackage predictions into numpy arrays
        
        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]
        
        if not do_unc:
            unc_unpacked = None
        if not do_deriv:
            deriv_unpacked = None
            
        return predict_unpacked, unc_unpacked, deriv_unpacked
        
    def __str__(self):
        """
        Returns a string representation of the model
        
        :returns: A string representation of the model (indicates number of sub-components
                  and array shapes)
        :rtype: str
        """
        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.get_n_emulators())+" emulators\n"+
                 str(self.get_n())+" training examples\n"+
                 str(self.get_D())+" input variables")
        