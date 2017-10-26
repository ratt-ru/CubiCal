# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

import logging
import os
import multiprocessing

# global verbosity level
verbosity = 0

# application name, used as the base of the logger
app_name = "app"

class NullWriter(object):
    """A null writer ignores messages"""
    def write(self, message):
        pass

_null_writer = NullWriter()

class LoggerWriter:
    def __init__(self, logger, level, verbose=None):
        self.logger = logger
        self.level = level
        self.verbose = verbose

    def verbosity(self, set_verb=None):
        global verbosity
        if set_verb is not None:
            self.verbose = set_verb
        return self.verbose if self.verbose is not None else verbosity

    def __call__(self, level):
        """
        Function call operator on logger. Use to issue messages at different verbosity levels.
        E.g. print>>log(2),"message" will only print message if the verbosity level is set to 2 or higher.

        Returns:
            Either the LoggerWriter itself (if level <= verbosity level), or a null writer which ignores messages.
        """
        # effective verbosity level is either set explicitly when the writer is created, or else use global level
        if level <= self.verbosity():
            return self
        else:
            return _null_writer

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)


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

log_memory = False
file_handler = None

def enableMemoryLogging (enable=True):
    global log_memory
    log_memory = enable

def logToFile (filename,append=False):
    global file_handler
    if not file_handler:
        file_handler = logging.FileHandler(filename, mode='a' if append else 'w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(M._formatter)
    logging.getLogger('').addHandler(file_handler)

def getLogFilename():
    '''Returns log filename if logToFile has been called previously, None otherwise'''
    global file_handler
    if not file_handler:
        return None
    return file_handler.baseFilename

class LoggerMemoryFilter (logging.Filter):
    def filter(self, event):
        vss = float(_memory()/(1024**3))
        vss_peak = float(_memory_peak()/(1024**3))
        rss = float(_resident()/(1024**3))
        rss_peak = float(_resident_peak()/(1024**3))
        shm = float(_shmem_size()/(1024**3))
        setattr(event,"virtual_memory_gb",vss)
        setattr(event,"resident_memory_gb",rss)
        setattr(event,"shared_memory_gb",shm)
        if log_memory and hasattr(event,"msg"):
            event.msg = "[%.1f/%.1f %.1f/%.1f %.1fGb] "%(rss,rss_peak,vss,vss_peak,shm) + event.msg
        subprocess_id = multiprocessing.current_process().name
        if subprocess_id != "MainProcess":
            subprocess_id = subprocess_id.replace("Process-", "P")
            event.msg = ModColor.Str("[%s] "%subprocess_id, col="blue") + event.msg
        return True


# global verbosity level. Only messages issued at this level and below will be printed.
verbosity = 0

class MyLogger():
    def __init__(self):
#       fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        fmt = " - %(asctime)s - %(name)-18.18s | %(message)s"
#        fmt = "%(asctime)s %(name)-25.25s | %(message)s"
        datefmt = '%H:%M:%S'#'%H:%M:%S.%f'
        logging.basicConfig(level=logging.DEBUG,format=fmt,datefmt=datefmt)
        self.Dico={}
        self.Silent=False
        self._myfilter = LoggerMemoryFilter()
        self._formatter = logging.Formatter(fmt, datefmt)


    def getLogger(self, name, verbose=None):

        if name not in self.Dico:
            logger = logging.getLogger("{}.{}".format(app_name, name))
            logger.addFilter(self._myfilter)
            self.Dico[name] = LoggerWriter(logger, logging.INFO, verbose)

        return self.Dico[name]




    #logger2 = logging.getLogger("demo.X")
    #debug_fp = LoggerWriter(logger2, logging.DEBUG)
    #print>>fp, ModColor.Str("An INFO message")
    #print >> debug_fp, "A DEBUG message"
    #print >> debug_fp, 1

M=MyLogger()

getLogger=M.getLogger

itsLog=getLogger("MyLogger")
import ModColor
def setSilent(Lname):
    print>>itsLog, ModColor.Str("Set silent: %s" % Lname, col="red")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.CRITICAL)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    print>>itsLog, ModColor.Str("Set loud: %s" % Lname, col="green")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.DEBUG)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    print>>log, "a.x"
