from abc import ABCMeta, abstractmethod, abstractproperty

from cubical import param_db
import numpy as np
from numpy.ma import masked_array

from cubical.tools import logger, ModColor
log = logger.getLogger("gain_machine")


class MasterMachine(object):
    """
    This is a base class for all solution machines. It is completely generic and lays out the basic
    requirements for all machines.
    
    It also provides a Factory class that takes care of creating machines and interfacing to solution
    tables on disk
    """

    __metaclass__ = ABCMeta

    def __init__(self, label, times, freqs, options):
    	"""
    	The init method of the overall abstract machine should know about the times and frequencies 
    	associated with its gains.
    	"""
        self.label = label
    	self.times = times
    	self.freqs = freqs
        self.options = options

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

    # dict of {label: (empty_value, axes)} describing the types of solutions that this machine exports.
    # Axes is a list of axis labels.
    # If empty_value is float or complex, global precision settings will be used
    exportable_solutions = {}

    # list of labels of solutions that this machine can import. This can be a subset of, or all of,
    # exportable_solutions.keys()
    importable_solutions = []

    @abstractmethod
    def get_solutions_grid(self):
        """
        Returns grid on which solutions are defined, e.g. {'time': vector, 'freq': vector}
        Note that axes which are fully spanned (e.g. antenna, correlation) need not be present in the grid.
        """
        return NotImplementedError

    @abstractmethod
    def export_solutions(self):
        """This method returns the solutions as a dict of {label: masked_array} elements.
        Array are masked since solutions have flags on them.
        Labels must be in exportable_solutions.
        """
        return NotImplementedError

    @abstractmethod
    def import_solutions(self, solutions_dict):
        """This method loads the internal solutions from a dict of {label: array} elements
        Labels must be in importable_solutions. 
        Array are masked since solutions have flags on them.
        Arrays shapes will conform to get_solutions_grid() results.
        """
        return NotImplementedError


    @classmethod
    def create_factory(machine_cls, *args, **kw):
        """This method creates a Machine Factory that will initialize the correct type of gain machine.

        This must be called via the subclasses, so that the factory gets the proper class information
        """
        return machine_cls.Factory(machine_cls, *args, **kw)



    class Factory(object):
        """
        A Machine Factory class is created as a singleton: it is responsible for creating and initializing 
        individual gain machines for chunks of the solution space. The purpose of a factory is to hold "global" 
        information for the solution overall (e.g. the overall time/frequency range etc.)
        
        TODO: directions must also be passed in
        """

        def __init__(self, machine_cls, grid, double_precision, apply_only, global_options, solution_options):
            """
            Initializes a factory

            Args:
                machine_cls: the class of the gain machine that will be created
                
                grid: the grid on which solutions are expected. Contains dict of 
                'time', 'freq', 'ant', 'corr' vectors.
            """
            self.global_options = global_options
            self.options = solution_options
            self.label = self.options['label']
            self.machine_class = machine_cls
            self.apply_only = apply_only
            self.grid = grid
            self.n_tim, self.n_fre, self.n_ant, self.n_cor = \
                [ len(grid.get(x)) for x in "time", "freq", "ant", "corr" ]
            self._ctype = np.complex128 if double_precision else np.complex64
            self._ftype = np.float64    if double_precision else np.float32
            # initialize solution databases
            self.init_solutions()

        def _make_filename(self, key):
            filename = self.options[key]
            try:
                return filename.format(**self.global_options)
            except Exception, exc:
                print>> log,"{}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc())
                print>>log,ModColor.Str("Error parsing {} filename '{}', see above".format(key, filename))
                raise ValueError(filename)

        def init_solutions(self):
            """Initializes solution databases"""
            # init solutions from database
            filename = self._make_filename("load-from")
            if filename:
                print>>log(0),ModColor.Str("{} solutions will be initialized from {}".format(self.label, filename), col="green")
                self._init_sols = param_db.load(filename)
            else:
                self._init_sols = None
            # create database to save to
            filename = self._make_filename("save-to")
            if not self.apply_only and filename:
                self._save_sols = param_db.create(filename, backup=True)
                for sol_label, (empty_value, axes) in self.machine_class.exportable_solutions.iteritems():
                    if type(empty_value) is float:
                        dtype = self._ftype
                    elif type(empty_value) is complex:
                        dtype = self._ctype
                    else:
                        dtype = type(empty_value)
                    self._save_sols.define_param("{}:{}".format(self.label, sol_label), dtype, axes,
                                                 empty=empty_value,
                                                 interpolation_axes=["time", "freq"])
                print>> log(0), "{} solutions will be saved to {}".format(self.label, filename)
            else:
                self._save_sols = None

        def export_solutions(self, gm, subdict):
            """Exports a slice of solutions from a gain machine into a shared dictionary"""
            if self.apply_only:
                return
            # populate grids subdictionary
            subdict["{}:grid__".format(self.label)] = gm.get_solutions_grid()
            # populate values subdictionary
            for sol_label, value in gm.export_solutions().iteritems():
                subdict["{}:{}".format(self.label, sol_label)] = value.data
                subdict["{}:{}:flags__".format(self.label, sol_label)] = value.mask

        def save_solutions(self, subdict):
            """Saves a slice of the solutions from a dictionary to the database"""
            if self.apply_only:
                return
            if self._save_sols is not None:
                # convert shared dict to regular dict for saving
                sd = subdict["{}:grid__".format(self.label)]
                grids = { key: sd[key] for key in sd.iterkeys() }
                # add slices for all parameters
                for name in subdict.iterkeys():
                    if not name.endswith("__"):
                        self._save_sols.add_chunk(name, masked_array(subdict[name], subdict[name+":flags__"]), grids)

        def close(self):
            if self._init_sols is not None:
                self._init_sols.close()
            if self._save_sols is not None:
                self._save_sols.close()

        def create_machine(self, data_arr, n_dir, n_mod, times, freqs):
            """
            Creates a gain machine, given a model, and a time and frequency subset of the global solution space.

            Args:
                model_arr: model array, of shape (ndir,nmod,ntim,nfreq,nant,nant,ncorr,ncorr)
                times: times
                freqs: frequencies

            Returns:
                An instance of a gain machine
            """
            gm = self.machine_class(self.label, data_arr, n_dir, n_mod, times, freqs, self.options)
            if self._init_sols is not None:
                sols = {}
                grid = gm.get_solutions_grid()
                # collect importable solutions from DB, interpolate
                for sol_label in gm.importable_solutions:
                    label = "{}:{}".format(self.label, sol_label)
                    if label in self._init_sols:
                        sols[sol_label] = self._init_sols[label].reinterpolate(**grid)
                # if anything at all was found in DB, import
                if sols:
                    gm.import_solutions(sols)
            return gm

