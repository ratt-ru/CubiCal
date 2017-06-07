from abc import ABCMeta, abstractmethod

class MasterMachine(object):
    """
    This is a base class for all solution machines. It is comletely generic and lays out the basic
    requirements for all machines.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def compute_js(self):
        return NotImplementedError

    @abstractmethod
    def compute_update(self):
        return NotImplementedError

    @abstractmethod
    def compute_residual(self):
        return NotImplementedError

    @abstractmethod
    def apply_inv_gains(self):
        return NotImplementedError

    @abstractmethod			
    def apply_gains(self):
        return NotImplementedError

    @abstractmethod				
    def compute_stats(self):
        return NotImplementedError

    @abstractmethod				
    def compute_chi_squared(self):
        return NotImplementedError

    def precompute_attributes(self, *args, **kwargs):
        return
