# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

import logging, os, re, sys, multiprocessing
import ModColor

# global verbosity level (used for loggers for which an explicit level is not set)
_global_verbosity = 0

class _NullWriter(object):
    """A null writer ignores messages"""
    def write(self, message):
        pass

class _DefaultWriter(object):
    """A default writer logs messages to a logger"""
    def __init__(self, logger, level, color=None, bold=None):
        self.logger = logger
        self.level = level
        self.color = (color or "red") if bold else color
        self.bold = bool(color) if bold is None else bold

    def write(self, message):
        if message != '\n':
            if self.color:
                message = ModColor.Str(message, col=self.color, Bold=self.bold)
            self.logger.log(self.level, message)

_null_writer = _NullWriter()

class LogWriter(object):
    def __init__(self, logger, level, verbose=None):
        self.logger = logger
        self.level = level
        self.verbose = verbose
        self._default_writer = _DefaultWriter(logger, level)

    def verbosity(self, set_verb=None):
        global _global_verbosity
        if set_verb is not None:
            self.verbose = set_verb
        return self.verbose if self.verbose is not None else _global_verbosity

    def __call__(self, level, color=None):
        """
        Function call operator on logger. Use to issue messages at different verbosity levels.
        E.g. print>>log(2),"message" will only print message if the verbosity level is set to 2 or higher.
        An optional color argument will colorize the message.

        Returns:
            A writer object (to which a message may be sent with "<<")
        """
        # effective verbosity level is either set explicitly when the writer is created, or else use global level
        if level <= self.verbosity():
            if color:
                return _DefaultWriter(self.logger, self.level, color=color)
            else:
                return self._default_writer
        else:
            return _null_writer

    def write(self, message):
        return self._default_writer.write(message)


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey,statusfile=None):
    global _scale
    _proc_status = '/proc/%d/status' % os.getpid()
    # get pseudo file  /proc/<pid>/status
    try:
        t = open(statusfile or _proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def _shmem_size (since=0.0):
    '''Return shared memory usage in bytes.'''
    return _VmB('Shmem:','/proc/meminfo') - since

def _memory(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmSize:') - since

def _memory_peak(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmPeak:') - since

def _resident(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmRSS:') - since

def _resident_peak(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmHWM:') - since

def _stacksize(since=0.0):
    '''Return stack size in bytes.'''
    return _VmB('VmStk:') - since

_file_handler = None

def logToFile(filename, append=False):
    global _file_handler
    if not _file_handler:
        _file_handler = logging.FileHandler(filename, mode='a' if append else 'w')
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(_logfile_formatter)
    _root_logger.addHandler(_file_handler)

def getLogFilename():
    '''Returns log filename if logToFile has been called previously, None otherwise'''
    global _file_handler
    if not _file_handler:
        return None
    return _file_handler.baseFilename

_log_memory = False
def enableMemoryLogging(enable=True):
    global _log_memory
    _log_memory = enable

class LogFilter(logging.Filter):
    """LogFilter augments the event by a few new attributes used by our formatter"""
    def filter(self, event):
        # short logger name (without app_name in front of it)
        setattr(event, 'shortname', event.name.split('.',1)[1] if '.' in event.name else event.name)
        setattr(event, 'separator', '| ')
        # memory usage info
        vss = float(_memory()/(1024**3))
        vss_peak = float(_memory_peak()/(1024**3))
        rss = float(_resident()/(1024**3))
        rss_peak = float(_resident_peak()/(1024**3))
        shm = float(_shmem_size()/(1024**3))
        setattr(event,"virtual_memory_gb",vss)
        setattr(event,"resident_memory_gb",rss)
        setattr(event,"shared_memory_gb",shm)
        if _log_memory:
            setattr(event, "memory", "[%.1f/%.1f %.1f/%.1f %.1fGb] "%(rss,rss_peak,vss,vss_peak,shm))
            setattr(event, 'separator', '')
        else:
            setattr(event, "memory", "")
        # subprocess info
        subprocess_id = multiprocessing.current_process().name
        if subprocess_id != "MainProcess":
            subprocess_id = subprocess_id.replace("Process-", "P")
            setattr(event, "subprocess", ModColor.Str("[%s] "%subprocess_id, col="blue"))
            setattr(event, 'separator', '')
        else:
            setattr(event, "subprocess", "")
        return True

_log_filter = LogFilter()

class ColorStrippingFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, strip=True):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.strip = strip

    def format(self, record):
        """Uses parent class to format record, then strips off colors"""
        msg = logging.Formatter.format(self, record)
        if self.strip:
            return re.sub("\033\\[[0-9]+m", "", msg, 0)
        else:
            return msg


_fmt = " - %(asctime)s - %(shortname)-18.18s %(subprocess)s%(memory)s%(separator)s%(message)s"
#        _fmt = "%(asctime)s %(name)-25.25s | %(message)s"
_datefmt = '%H:%M:%S'#'%H:%M:%S.%f'
_logfile_formatter = ColorStrippingFormatter(_fmt, _datefmt, strip=True)
_console_formatter = ColorStrippingFormatter(_fmt, _datefmt, strip=False)

_root_logger = None
log = None

def init(app_name):
    global _app_name
    global _root_logger
    if _root_logger is None:
        logging.basicConfig(level=logging.DEBUG, fmt=_fmt, datefmt=_datefmt)
        _app_name = app_name
        _root_logger = logging.getLogger(app_name)
        _root_logger.propagate = False
        _console_handler = logging.StreamHandler(sys.stderr)
        _console_handler.setFormatter(_console_formatter)
        _console_handler.setLevel(logging.DEBUG)
        _root_logger.addFilter(_log_filter)
        _root_logger.addHandler(_console_handler)
        global log
        log = _loggers[''] = LogWriter(_root_logger, logging.INFO)

# dict of LogWriters
_loggers = {}

def getLogger(name, verbose=None):
    """Creates a new logger (or returns one, if already created)"""
    init("app")
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger("{}.{}".format(_app_name, name))
    logger.addFilter(_log_filter)
#    logger.propagate = True
    lw = _loggers[name] = LogWriter(logger, logging.INFO, verbose)
    print>>lw(2), "logger initialized"

    return lw

def setBoring(boring=True):
    _console_formatter.strip = boring

def setVerbosity(verbosity):
    if type(verbosity) is int:
        _global_verbosity = verbosity
        if _global_verbosity:
            print>>log(0, "green"), "set global verbosity level {}".format(_global_verbosity)
    elif type(verbosity) is str and re.match("^[0-9]+$", verbosity):
        _global_verbosity = int(verbosity)
        if _global_verbosity:
            print>>log(0, "green"), "set global verbosity level {}".format(_global_verbosity)
    else:
        if isinstance(verbosity, str):
            verbosity = verbosity.split(",")
        elif not isinstance(verbosity, (list, tuple)):
            raise TypeError("can't parse verbosity specification of type '{}'".format(type(verbosity)))
        for element in verbosity:
            m = re.match("^(.+)=([0-9]+)$", element)
            if not m:
                raise ValueError("can't parse verbosity specification '{}'".format(element))
            logger = getLogger(m.group(1))
            verbosity = int(m.group(2))
            logger.verbosity(verbosity)
            print>>logger(0,"green"),"set verbosity level {}".format(verbosity)

def setSilent(Lname):
    """Silences the specified sublogger(s)"""
    print>>log, ModColor.Str("set silent: %s" % Lname, col="red")
    if type(Lname) is str:
        getLogger(Lname).logger.setLevel(logging.CRITICAL)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    """Un-silences the specified sublogger(s)"""
    print>>log, ModColor.Str("set loud: %s" % Lname, col="green")
    if type(Lname) is str:
        getLogger(Lname).logger.setLevel(logging.DEBUG)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    print>>log, "a.x"
