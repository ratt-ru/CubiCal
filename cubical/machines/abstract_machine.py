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
    def compute_stats(self):
    	"""
    	This method should compute a variety of useful parameters regarding the conditioning and 
    	degrees of freedom of the current time-frequency chunk. Specifically, it must populate 
    	an attribute containing the degrees of freedom per time-frequency slot. 
    	"""
        return NotImplementedError

    @abstractmethod				
    def is_converged(self):
    	"""
    	This method should check the convergence of the current time-frequency chunk. Should return 
    	a Boolean.
    	"""
        return NotImplementedError

    @abstractmethod				
    def compute_chi_squared(self):
    	"""
    	Ignore for now - will likey form part of the Jones Chain.
    	"""
        return NotImplementedError

    def precompute_attributes(self, *args, **kwargs):
        return
