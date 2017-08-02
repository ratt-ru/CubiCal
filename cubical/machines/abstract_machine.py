from abc import ABCMeta, abstractmethod

class MasterMachine(object):
    """
    This is a base class for all solution machines. It is comletely generic and lays out the basic
    requirements for all machines.
    """

    __metaclass__ = ABCMeta

    def __init__(self, times, freqs):
    	"""
    	The init method of the overall abstract machine should know about the times and frequencies 
    	associated with its gains.
    	"""
    	self.times = times
    	self.freqs = freqs

    @staticmethod
    def get_parameter_axes():
        """Returns list of axes over which a parameter is defined"""
        return []

    @abstractmethod
    def get_solution_slice():
        """Returns list of axes over which a parameter is defined"""
        return []


    @abstractmethod
    def compute_js(self):
    	"""
		This method is expected to compute (J^HJ)^-1 and J^HR. In practice, this method can be 
		very flexible, as it is only used in the compute_update method and need only be consistent
		with that usage. Should support the use of both the true residual and the observed data. 
		"""
        return NotImplementedError

    @abstractmethod
    def compute_update(self):
    	"""
    	This method is expected to compute the parameter update. As such, it must fetch or compute 
    	the terms of the update in order to update the gains. Should call the compute_js but is 
    	very flexible, provided it ultimately updates the gains. 
    	"""
        return NotImplementedError

    @abstractmethod
    def compute_residual(self):
    	"""
    	This method should compute the residual at the the full time-frequency resolution of the
    	data. Should return the residual.
    	"""
        return NotImplementedError

    @abstractmethod
    def apply_inv_gains(self):
    	"""
    	This method should be able to apply the inverse of the gains to an array at full time-
    	frequency resolution. Should return the input array at full resolution after the application
    	of the inverse gains.
    	"""
        return NotImplementedError

    @abstractmethod			
    def apply_gains(self):
    	"""
    	This method should be able to apply the gains to an array at full time-frequency
    	resolution. Should return the input array at full resolution after the application of the 
    	gains.
    	"""
        return NotImplementedError

    @abstractmethod				
    def update_stats(self):
    	"""
    	This method should compute a variety of useful parameters regarding the conditioning and 
    	degrees of freedom of the current time-frequency chunk. Specifically, it must populate 
    	an attribute containing the degrees of freedom per time-frequency slot. 
    	"""
        return NotImplementedError

    @abstractmethod				
    def update_conv_params(self):
    	"""
    	This method should check the convergence of the current time-frequency chunk. Should return 
    	a Boolean.
    	"""
        return NotImplementedError

    def precompute_attributes(self, *args, **kwargs):
        return

    @classmethod
    def create_factory(machine_cls, times, freqs, antennas, correlations, options):
        """This method creates a Machine Factory that will initialize the correct type of gain machine.

        This must be called via the subclasses, so that the factory gets the proper class information
        """
        return machine_cls.Factory(machine_cls, times, freqs, antennas, correlations, options)

    class Factory(object):
        """
        A Machine Factory class is created as a singleton: and it is responsible for creating and initializing 
        individual gain machines for chunks of the solution space. The purpose of a factory is to hold "global" 
        information for the solution overall (e.g. the overall time/frequency range etc.)
        
        TODO: directions must also be passed in
        """

        def __init__(self, machine_cls, times, freqs, antennas, correlations, options):
            """
            Initializes a factory

            Args:
                machine_cls: the class of the gain machine that will be created
            """
            self.machine_class = machine_cls
            self.times, self.freqs, self.antennas, self.correlations, self.options = \
                times, freqs, antennas, correlations, options
            self.n_tim, self.n_fre, self.n_ant, self.n_cor = \
                [len(x) for x in times, freqs, antennas, correlations]

        def create(self, model_arr, times, freqs):
            """
            Creates a gain machine, given a model, and a time and frequency subset of the global solution space.

            Args:
                model_arr: model array, of shape (ndir,nmod,ntim,nfreq,nant,nant,ncorr,ncorr)
                times: times
                freqs: frequencies

            Returns:
                An instance of a gain machine
            """
            n_dir, n_mod = model_arr.shape[:2]
            assert (model_arr.shape == (n_dir, n_mod, len(times), len(freqs),
                                        self.n_ant, self.n_ant, self.n_cor, self.n_cor))
            return self.machine_class(model_arr, times, freqs, self.options)

