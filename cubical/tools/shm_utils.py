# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet
from __future__ import print_function
import os, re, errno
from . import logger
from . import NpShared
from . import shared_dict

log = logger.getLogger("shm_utils")

# store PID here, so that it corresponds to the PID of the parent thread (when we fork off child processes)
_pid = os.getpid()

program_name = "cubical"

def getShmPrefix():
    """Returns prefix used for shared memory arrays. ddf.PID is the convention"""
    return "{}.{}".format(program_name, _pid)

# init SharedDict with the same base name
shared_dict.SharedDict.setBaseName(getShmPrefix())

def getShmName(name, **kw):
    """
    Forms up a name for a shm-backed shared element. This takes the form of "ddf.PID.", where PID is the
    pid of the process where the cache manager was created (so the parent process, presumably), followed
    by a filename of the form "NAME:KEY1_VALUE1:...", as returned by getElementName(). See getElementName()
    for usage.
    """
    # join keyword args into "key=value:key=value:..."
    kws = ":".join([name] + ["%s_%s" % (key, value) for key, value in sorted(kw.items())])
    return "%s.%s" % (getShmPrefix(), kws)

def cleanupShm ():
    """
    Deletes all shared arrays for this process
    """
    NpShared.DelAll(getShmPrefix())
    # above statement don't work for directories and subdirectories
    os.system("rm -rf /dev/shm/%s"%getShmPrefix())

def cleanupStaleShm ():
    """
    Cleans up "stale" shared memory from previous runs of CubeCal
    """
    # check for stale shared memory
    uid = os.getuid()
    # list of all files in /dev/shm/ matching ddf.PID.* and belonging to us
    shmlist = [ ("/dev/shm/"+filename, re.match('(sem\.)?' + program_name + '\.([0-9]+)(\..*)?$',filename)) for filename in os.listdir("/dev/shm/")
                if os.stat("/dev/shm/"+filename).st_uid == uid ]
    # convert to list of filename,pid tuples
    shmlist = [ (filename, int(match.group(2))) for filename, match in shmlist if match ]
    # now check all PIDs to find dead ones
    # if we get ESRC error from sending signal 0 to the process, it's not running, so we mark it as dead
    dead_pids = set()
    for pid in set([x[1] for x in shmlist]):
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:
                dead_pids.add(pid)
    # ok, make list of candidates for deletion
    victims = [ filename for filename,pid in shmlist if pid in dead_pids ]
    if victims:
        print("reaping %d shared memory objects associated with %d dead cubical processes"%(len(victims), len(dead_pids)), file=log)
        dirs = [ v for v in victims if os.path.isdir(v) ]
        files = [ v for v in victims if not os.path.isdir(v) ]
        # rm -fr only works for a limited number of arguments (which the semaphore list can easily exceed)
        # so use os.unlink() to remove files, and rm -fr for directories
        for path in files:
            os.unlink(path)
        os.system("rm -fr " + " ".join(dirs))
        # print "rm -fr " + " ".join(victims)

