# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

from __future__ import print_function
from six import string_types
import logging, logging.handlers, os, re, sys, multiprocessing, time, signal
import psutil
from . import ModColor

# dict of logger wrappers created by the application
_loggers = {}

# global verbosity levels (used for loggers for which an explicit level is not set)
_global_verbosity = 0
_global_log_verbosity = None

# this will be the handler for the log file
_file_handler = None
# this will be a null handler
_null_handler = logging.NullHandler()

def logToFile(filename, append=False):
    global _file_handler
    if not _file_handler:
        _file_handler = logging.FileHandler(filename, mode='a' if append else 'w')
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(_logfile_formatter)
        # set it as the target for the existing wrappers' handlers
        for wrapper in _loggers.values():
            wrapper.logfile_handler.setTarget(_file_handler)

def getLogFilename():
    '''Returns log filename if logToFile has been called previously, None otherwise'''
    global _file_handler
    if not _file_handler:
        return None
    return _file_handler.baseFilename

class _DefaultWriter(object):
    """A default writer logs messages to a logger"""
    __print_once_keys = []
    def __init__(self, logger, level, color=None, bold=None):
        self.logger = logger
        self.level = level
        self.color = (color or "red") if bold else color
        self.bold = bool(color) if bold is None else bold

    def write(self, message, level_override=None, print_once=None):
        if print_once is not None:
            if print_once in _DefaultWriter.__print_once_keys:
                return
            _DefaultWriter.__print_once_keys = list(set(_DefaultWriter.__print_once_keys + [print_once]))

        message = message.rstrip()
        if self.color and message:  # do not colorize empty messages, else "\n" is issued independently
            message = ModColor.Str(message, col=self.color, Bold=self.bold)
        self.logger.log(self.level if level_override is None else level_override, message)

    def print(self, *args):
        return self.write(" ".join(map(str, args)))

class LoggerWrapper(object):
    def __init__(self, logger, verbose=None, log_verbose=None):
        self.logger = logger
        logger.propagate = False

        # initialize handlers for console and logfile

        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setFormatter(_console_formatter)

        self.logfile_handler = logging.handlers.MemoryHandler(1, logging.DEBUG, _file_handler or _null_handler)
        self.logfile_handler.setFormatter(_logfile_formatter)

        # set verbosity levels
        self._verbose = self._log_verbose = None
        self.verbosity(verbose if verbose is not None else _global_verbosity)
        self.log_verbosity(log_verbose if log_verbose is not None else _global_log_verbosity)

        # other init

        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.logfile_handler)
        self.logger.addFilter(_log_filter)

    def verbosity(self, set_verb=None):
        if set_verb is not None:
            self._verbose = set_verb
            self.console_handler.setLevel(logging.INFO - set_verb)
            if self._log_verbose is None:
                self.logfile_handler.setLevel(logging.INFO - set_verb)
        return self._verbose

    def log_verbosity(self, set_verb=None):
        if set_verb is not None:
            self._log_verbose = set_verb
            self.logfile_handler.setLevel(logging.INFO - set_verb)
        return self._log_verbose if self._log_verbose is not None else self._verbose

    def __call__(self, level, color=None):
        """
        Function call operator on logger. Use to issue messages at different verbosity levels.
        E.g. log(2).print("message" will issue a message at level logging.INFO - 2.)
        An optional color argument will colorize the message.

        Returns:
            A writer object (to which a message may be sent with "<<")
        """
        # effective verbosity level is either set explicitly when the writer is created, or else use global level
        return _DefaultWriter(self.logger, logging.INFO - level, color=color)
    
    def warn(self, msg, color=None, print_once=None):
        """
        Wrapper for log.warn
        """
        _DefaultWriter(self.logger, logging.WARN, color=color).write(msg, print_once=print_once)

    warning = warn

    def error(self, msg, color="red", print_once=None):
        """
        Wrapper for log.error
        """
        _DefaultWriter(self.logger, logging.ERROR, color=color).write(msg, print_once=print_once)
        
    def info(self, msg, color=None, print_once=None):
        """
        Wrapper for log.info
        """
        _DefaultWriter(self.logger, logging.INFO, color=color).write(msg, print_once=print_once)

    def critical(self, msg, color=None, print_once=None):
        """
        Wrapper for log.critical
        """
        _DefaultWriter(self.logger, logging.CRITICAL, color=color).write(msg, print_once=print_once)

    def debug(self, msg, color=None, print_once=None):
        """
        Wrapper for log.debug
        """
        _DefaultWriter(self.logger, logging.DEBUG, color=color).write(msg, print_once=print_once)

    def exception(self, msg, color=None, print_once=None):
        """
        Wrapper for log.exception
        """
        _DefaultWriter(self.logger, logging.EXCEPTION, color=color).write(msg, print_once=print_once)

    def print(self, *args):
        return self.info(" ".join(map(str, args)))

    def write(self, message, level=logging.INFO, verbosity=0, print_once=None, color=None):
        # apply verbosity only to INFO levels
        if level == logging.INFO:
            level -= int(verbosity)
        _DefaultWriter(self.logger, level, color=color).write(message,
                                                                               print_once=print_once)

_parent_process = psutil.Process(os.getpid())
_log_memory_totals = True
_log_memory_types = None, "rss vms".split(), "rss pss".split()
GB = float(1024 ** 3)


def enableMemoryLogging(level=1):
    LogFilter.setMemoryLogging((level or 0)%3)   # level is 0/1/2

_subprocess_label = None

def set_subprocess_label(label):
    """
    Sets the subprocess label explicitly
    """
    global _subprocess_label
    _subprocess_label = label


def get_subprocess_label():
    """
    Returns the subprocess ID. For the main process, this is empty. For subprocesses
    (started by multiprocessing), this is "Pn" by default, where n is the process number.
    """
    global _subprocess_label
    if _subprocess_label is None:
        name = multiprocessing.current_process().name
        if name == "MainProcess":
            _subprocess_label = ""
        else:
            _subprocess_label = name.replace("Process-", "P")
    return _subprocess_label


def _sigusr1_handler(signum, frame):
    level = 2 if LogFilter._log_memory == 1 else 1
    print("pid {} received USR1: memory logging level {}".format(os.getpid(), level))
    LogFilter.setMemoryLogging(level)


def _sigusr2_handler(signum, frame):
    print("pid {} received USR2: disabling memory logging".format(os.getpid()))
    LogFilter.setMemoryLogging(0)


signal.signal(signal.SIGUSR1, _sigusr1_handler)
signal.signal(signal.SIGUSR2, _sigusr2_handler)

class LogFilter(logging.Filter):
    _log_memory = 0


    @staticmethod
    def setMemoryLogging(level):
        LogFilter._log_memory = level
        LogFilter._mem_ts = LogFilter._mem_totals_ts = LogFilter._children_ts = 0

    _mem = None
    _mem_totals = None
    _children = None
    _mem_ts = 0
    _mem_totals_ts = 0
    _children_ts = 0
    _mem_update = (0, .5, 2)     # full memory updates are a little costly, so do them only after this many seconds has elapsed
    _mem_totals_update = (0, 1, 5)
    _children_update = 3 # children updates even more so, so do it even less frequently

    """LogFilter augments the event by a few new attributes used by our formatter"""
    def filter(self, event):
        if not event.getMessage().strip():
            return False
        # short logger name (without app_name in front of it)
        setattr(event, 'shortname', event.name.split('.',1)[1] if '.' in event.name else event.name)
        setattr(event, 'separator', '| ')
        memlevel = LogFilter._log_memory  # signal handler can change that midway through, so use this value
        if memlevel:
            # memory usage info
            t = time.time()
            if t - self._mem_ts > self._mem_update[memlevel]:
                KEYS = _log_memory_types[memlevel]
                # get memory for this process
                mi0 = psutil.Process(os.getpid()).memory_full_info() if memlevel == 2 else \
                    psutil.Process(os.getpid()).memory_info()
                mem = {key: getattr(mi0, key) / GB for key in KEYS}
                shm = psutil.virtual_memory().shared / GB
                # get total memory
                if _log_memory_totals:
                    if t - self._mem_totals_ts > self._mem_totals_update[memlevel]:
                        if t - self._children_ts > self._children_update:
                            self._children = [_parent_process] + list(_parent_process.children(recursive=True))
                            self._children_ts = time.time()
                        if len(self._children) > 1:
                            mis = []
                            # scan over children, ignoring ones that may have disappeared
                            for p in self._children:
                                try:
                                    mis.append(p.memory_full_info()) if memlevel == 2 else mis.append(p.memory_info())
                                except psutil.NoSuchProcess:
                                    pass
                            self._mem_totals = {key: sum([getattr(mi,key) for mi in mis]) / GB for key in KEYS}
                        self._mem_totals_ts = time.time()
                # form up string
                if self._mem_totals is None:
                    smem = ["{:.1f}".format(mem.get(key, 0)) for key in KEYS]
                else:
                    smem = ["{:.1f}/{:.1f}".format(mem.get(key, 0), self._mem_totals.get(key, 0)) for key in KEYS]
                smem.append("{:.1f}Gb".format(shm))
                self._mem = " ".join(smem)
                self._mem_ts = time.time()
            setattr(event, "memory", "[{}{}] ".format("*" if memlevel == 2 else "", self._mem))
            setattr(event, 'separator', '')
        else:
            setattr(event, "memory", "")
        # subprocess info
        subprocess_id = get_subprocess_label()
        if subprocess_id:
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

    def label(self, record):
        if record.levelno < logging.WARNING:
            return "INFO", "37;42"
        elif record.levelno < logging.ERROR:
            return "WARNING", "37;43"
        elif record.levelno < logging.CRITICAL:
            return "ERROR", "5;41"
        else:
            return "CRITICAL", "5;41"


    def format(self, record):
        """Uses parent class to format record, then strips off colors"""
        msg = logging.Formatter.format(self, record)
        label, color = self.label(record)
        if self.strip:
            return "{:10s}{}".format(label, re.sub("\033\\[[0-9]+m", "", msg, 0))
        else:
            return "\033[1;{}m{:10s}\033[0m{}".format(color, label, msg)\

_fmt = "%(asctime)s - %(shortname)-18.18s %(subprocess)s%(memory)s%(separator)s%(message)s"
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
        # logging.basicConfig(level=logging.DEBUG, datefmt=_datefmt)
        logging.basicConfig(datefmt=_datefmt)
        _app_name = app_name
        _root_logger = logging.getLogger(app_name)
        _root_logger.setLevel(logging.DEBUG)
        global log
        log = _loggers[''] = LoggerWrapper(_root_logger)

def getLogger(name, verbose=None, log_verbose=None):
    """Creates a new logger (or returns one, if already created)"""
    init("app")
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger("{}.{}".format(_app_name, name))
    lw = _loggers[name] = LoggerWrapper(logger, verbose, log_verbose)
    lw(2).print("logger initialized")

    return lw

def setBoring(boring=True):
    _console_formatter.strip = boring

def setGlobalVerbosity(verbosity):
    global _global_verbosity
    if verbosity is None:
        _global_verbosity = 0
        return
    # ensure verbosity is turned into a list.
    if type(verbosity) is int:
        verbosity = [verbosity]
    elif isinstance(verbosity, string_types):
        verbosity = verbosity.split(",")
    elif not isinstance(verbosity, (list, tuple)):
        raise TypeError("can't parse verbosity specification of type '{}'".format(type(verbosity)))
    for element in verbosity:
        if type(element) is int or re.match("^[0-9]+$", element):
            _global_verbosity = int(element)
            log(0, "green").print("set global console verbosity level {}".format(_global_verbosity))
        else:
            m = re.match("^(.+)=([0-9]+)$", element)
            if not m:
                raise ValueError("can't parse verbosity specification '{}'".format(element))
            logger = getLogger(m.group(1))
            level = int(m.group(2))
            logger.verbosity(level)
            logger(0,"green").print("set console verbosity level {}={}".format(m.group(1), level))

def setGlobalLogVerbosity(verbosity):
    global _global_log_verbosity
    if verbosity is None:
        _global_log_verbosity = None  # None means follow console default
        return
    # ensure verbosity is turned into a list.
    if type(verbosity) is int:
        verbosity = [verbosity]
    elif isinstance(verbosity, string_types):
        verbosity = verbosity.split(",")
    elif not isinstance(verbosity, (list, tuple)):
        raise TypeError("can't parse verbosity specification of type '{}'".format(type(verbosity)))
    for element in verbosity:
        if type(element) is int or re.match("^[0-9]+$", element):
            _global_log_verbosity = int(element)
            if _global_log_verbosity is not None:
                log(0, "green").print("set global log verbosity level {}".format(_global_log_verbosity))
        else:
            m = re.match("^(.+)=([0-9]+)$", element)
            if not m:
                raise ValueError("can't parse verbosity specification '{}'".format(element))
            logger = getLogger(m.group(1))
            level = int(m.group(2))
            logger.log_verbosity(level)
            logger(0,"green").print("set log verbosity level {}={}".format(m.group(1), level))


def setSilent(Lname):
    """Silences the specified sublogger(s)"""
    log.print(ModColor.Str("set silent: %s" % Lname, col="red"))
    if isinstance(Lname, string_types):
        getLogger(Lname).logger.setLevel(logging.CRITICAL)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    """Un-silences the specified sublogger(s)"""
    log.print(ModColor.Str("set loud: %s" % Lname, col="green"))
    if isinstance(Lname, string_types):
        getLogger(Lname).logger.setLevel(logging.DEBUG)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    log.print("a.x")
