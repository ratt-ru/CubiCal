'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import os, re, errno
import logger
import NpShared
import shared_dict

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
        except OSError, err:
            if err.errno == errno.ESRCH:
                dead_pids.add(pid)
    # ok, make list of candidates for deletion
    victims = [ filename for filename,pid in shmlist if pid in dead_pids ]
    if victims:
        print>>log, "reaping %d shared memory objects associated with %d dead cubical processes"%(len(victims), len(dead_pids))
        dirs = [ v for v in victims if os.path.isdir(v) ]
        files = [ v for v in victims if not os.path.isdir(v) ]
        # rm -fr only works for a limited number of arguments (which the semaphore list can easily exceed)
        # so use os.unlink() to remove files, and rm -fr for directories
        for path in files:
            os.unlink(path)
        os.system("rm -fr " + " ".join(dirs))
        # print "rm -fr " + " ".join(victims)

