from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from numpy.ma import masked_array
import traceback

from cubical import param_db

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

    def __init__(self, label, data_arr, ndir, nmod, times, freqs, options):
        """
        The init method of the overall abstract machine should know about the times and frequencies 
        associated with its gains.
        """
        self.jones_label = label
        self.times = times
        self.freqs = freqs
        self.options = options
        self.solvable = options['solvable']

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

    @abstractmethod
    def flag_solutions(self):
        """
        This method should do solution flagging based on the gains.
        """
        return NotImplementedError

    @abstractmethod
    def propagate_gflags(self, flags):
        """
        This method should propagate the flags raised by the gain machine back into the data.
        This is necessary as the gain flags may not have the same shape as the data.
        """
        return NotImplementedError

    @abstractmethod
    def update_term(self):
        """
        This method should update the current iteration as well as handling any more complicated
        behaviour required for multiple Jones terms.
        """
        return NotImplementedError

    @abstractmethod
    def restrict_solution(self):
        """
        This method should perform any necessary restrictions on the solution, eg. selecting a 
        reference antenna or taking only the amplitude.
        """
        return NotImplementedError

    def precompute_attributes(self, *args, **kwargs):
        return

    @abstractproperty
    def has_converged(self):
        """
        This property must return the convergence status of the gain machine. 
        """
        return NotImplementedError

    # Returns dict of {label: (empty_value, axes_list)} describing the types of parameters that
    # this machine exports. Axes is a list of axis labels.
    # Static method, as it is called before any GM is actually created.
    # If empty_value is float or complex, global precision settings will be used.
    @staticmethod
    def exportable_solutions():
        return {}

    # Returns dict of parameters that this machine can import, as {label: grid_dict}
    # Grid_dict tells what grid the parameters must be interpolated onto for this machine.
    # Called when a machine has been created (so grids are available)
    def importable_solutions(self):
        return {}

    @abstractmethod
    def export_solutions(self):
        """This method returns the solutions as a dict of {label: masked_array, grid} elements.
        Array are masked since solutions have flags on them.
        Labels must be present in whatever exportable_solutions() returns.
        Grid is a dict, defining axes on which solutions are given, e.g. {'time': vector, 'freq': vector}
        Note that axes which are fully spanned (e.g. antenna, correlation) need not be present in the grid.
        """
        return NotImplementedError

    @abstractmethod
    def import_solutions(self, solutions_dict):
        """This method loads the internal solutions from a dict of {label: array} elements
        Labels must be in importable_solutions. 
        Array are masked since solutions have flags on them.
        Arrays shapes will conform to importable_solutions() results.
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

        def __init__(self, machine_cls, grid, double_precision, apply_only, global_options, jones_options):
            """
            Initializes a factory

            Args:
                machine_cls: the class of the gain machine that will be created
                
                grid: the grid on which solutions are expected, for axes that are already known (antennas, correlations)
                  A dictianary of {axis_name:values_vector}. time/freq need not be supplied, as it is filled in chunk by chunk.
                  
                double_precision: if True, gain machines will use 64-bit float arithmetic
                
                apply_only: if True, solutions are only applied, not solved for
                
                global_options: dict of global options for CubiCal (GD)
                
                jones_options: dict of Jones term options for the specific Jones term
            """
            self.global_options = global_options
            self.jones_options = jones_options
            self.jones_label = self.jones_options['label']
            self.machine_class = machine_cls
            if apply_only:
                self.jones_options['solvable'] = False
            self.solvable = self.jones_options['solvable']
            self.grid = grid
            self._ctype = np.complex128 if double_precision else np.complex64
            self._ftype = np.float64    if double_precision else np.float32
            # dict of jones label -> param database object to init from
            self._init_sols = {}
            # dict of jones label -> param database object to save to. Multiple jones labels
            # may share the same DB (_save_sols_byname holds a unique dict)
            self._save_sols = {}
            # dict of filename -> unique param database object.
            self._save_sols_byname = {}
            # initialize solution databases
            self.init_solutions()

        def init_solutions(self):
            """Internal method. Initializes solution databases. Note that this is reimplemented in JonesChain."""
            self._init_solutions(self.jones_label, self._make_filename(self.jones_options["load-from"]),
                                 self.solvable and self._make_filename(self.jones_options["save-to"]),
                                 self.machine_class.exportable_solutions())

        def _make_filename(self, filename):
            """Helper method: expands full filename from templated interpolation string"""
            try:
                return filename.format(**self.global_options)
            except Exception, exc:
                print>> log,"{}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc())
                print>>log,ModColor.Str("Error parsing {} filename '{}', see above".format(key, filename))
                raise ValueError(filename)

        def _init_solutions(self, label, load_from, save_to, exportables):
            """Internal helper method for init_solutions(): initializes a pair of solution databases"""
            # init solutions from database
            if load_from:
                print>>log(0),ModColor.Str("{} solutions will be initialized from {}".format(label, load_from), col="green")
                if "//" in load_from:
                    filename, prefix = load_from.rsplit("//", 1)
                else:
                    filename, prefix = load_from, label
                self._init_sols[label] = param_db.load(filename), prefix
            # create database to save to
            if save_to:
                db = self._save_sols_byname.get(save_to)
                if db is None:
                    self._save_sols_byname[save_to] = db = param_db.create(save_to, backup=True)
                self._save_sols[label] = db
                for sol_label, (empty_value, axes) in exportables.iteritems():
                    if type(empty_value) is float:
                        dtype = self._ftype
                    elif type(empty_value) is complex:
                        dtype = self._ctype
                    else:
                        dtype = type(empty_value)
                    db.define_param("{}:{}".format(label, sol_label), dtype, axes, empty=empty_value,
                                    interpolation_axes=["time", "freq"], grid=self.grid)
                print>> log(0), "{} solutions will be saved to {}".format(label, save_to)

        def export_solutions(self, gm, subdict):
            """Exports a slice of solutions from a gain machine into a shared dictionary.
            This is called in a solver process.
            """
            if not self.solvable:
                return
            sols = gm.export_solutions()
            # has the gain machine added a prefix to the names already (as the chain machine does)
            is_prefixed = sols.pop('prefixed', False)
            # populate values subdictionary
            for label, (value, grid) in sols.iteritems():
                name = label if is_prefixed else "{}:{}".format(gm.jones_label, label)
                subdict[name] = value.data
                subdict["{}:grid__".format(name)]  = grid
                subdict["{}:flags__".format(name)] = value.mask

        def save_solutions(self, subdict):
            """Saves a slice of the solutions from a dictionary to the database.
            This is called in an I/O process"""
            # add slices for all parameters
            for name in subdict.iterkeys():
                if not name.endswith("__"):
                    jones_label = name.split(':')[0]
                    if jones_label in self._save_sols:
                        sd = subdict["{}:grid__".format(name)]
                        grids = {key: sd[key] for key in sd.iterkeys()}
                        self._save_sols[jones_label].add_chunk(name, masked_array(subdict[name],
                                                                                  subdict[name+":flags__"]), grids)

        def close(self):
            for db, prefix in self._init_sols.values():
                db.close()
            for db in self._save_sols_byname.values():
                db.close()
            self._init_sols = {}
            self._save_sols = {}
            self._save_sols_byname = {}

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
            gm = self.machine_class(self.jones_label, data_arr, n_dir, n_mod, times, freqs, self.jones_options)
            sols = {}
            # collect importable solutions from DB, interpolate
            for label, grids in gm.importable_solutions().iteritems():
                db, prefix = self._init_sols.get(self.jones_label, (None, None))
                name = "{}:{}".format(prefix, label)
                if db is not None:
                    if name in db:
                        print>>log,"initializing {} using {} from {}".format(self.jones_label, name, db.filename)
                        sols[label] = db[name].reinterpolate(**grids)
                    else:
                        print>>log,"not initializing {}: {} not in {}".format(self.jones_label, name, db.filename)
            # if anything at all was loaded from DB, import
            if sols:
                gm.import_solutions(sols)
            return gm
