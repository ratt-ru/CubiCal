#!/usr/bin/python
from __future__ import print_function
import os, os.path, sys
from casacore.tables import table
import numpy as np
from collections import OrderedDict

def kw_to_args(**kw):
    """Converts a dict of keywords to a cubical command line. Makes sure --sol-jones comes first."""
    if "sol_jones" in kw:
        cmd = "--sol-jones {} ".format(kw.pop("sol_jones"))
    else:
        cmd = ""
    cmd += " ".join(["--{} {}".format(name.replace("_", "-"), value) for name, value in list(kw.items())])
    return cmd


basedir = os.path.dirname(__file__)

def logprint(arg):
    print(arg, file=sys.stderr)

class SolverVerification(object):
    def __init__(self, msname, refmsname, parset, workdir="."):
        self.msname = os.path.abspath(msname if "/" in msname else os.path.join(basedir,msname))
        self.refmsname = os.path.abspath(refmsname if "/" in refmsname else os.path.join(basedir,refmsname))
        self.parset = os.path.abspath(parset if "/" in parset else os.path.join(basedir,parset))
        self.cmdline = "gocubical {} ".format(self.parset)
        os.chdir(workdir)
        logprint("*** Working directory is {}".format(os.getcwd()))

    def generate_reference(self, colname, args=[], **kw):
        cmd = self.cmdline + kw_to_args(data_ms=self.refmsname, out_column=colname, out_name="ref_"+colname+"/cc", **kw) + \
                " " + " ".join(args)
        logprint("*** running {}".format(cmd))
        retcode = os.system(cmd)
        if retcode:
            raise RuntimeError("gocubical failed, return code {}".format(retcode))

    def verify(self, refcolname, args=[], mean_tolerance=-30, ninetyfifth_tolerance=-25, **kw):
        cmd = self.cmdline + kw_to_args(data_ms=self.msname, out_column="CORRECTED_DATA", out_name="test_"+refcolname+"/cc", **kw) + \
                " " + " ".join(args)
        logprint("*** running {}".format(cmd))
        retcode = os.system(cmd)
        if retcode:
            raise RuntimeError("{}: return code {}".format(cmd, retcode))
        cd = table(self.msname).getcol("CORRECTED_DATA")
        if not np.isfinite(cd).all():
            raise RuntimeError("{}: NaNs/INFs detected in output data".format(cmd))
        c0 = table(self.refmsname).getcol(refcolname)
        diff = abs(abs(cd-c0)/abs(c0))
        diffmean = 10*np.log10(np.nanmean(diff))
        logprint("*** mean relative diff between CORRECTED_DATA and {} is {} dB".format(refcolname, diffmean))
        if diffmean > mean_tolerance:
            raise RuntimeError("{}: diff {} dB exceeds tolerance of {} dB".format(cmd, diffmean, mean_tolerance))
        diff95 = 10*np.log10(np.nanpercentile(diff, 95.0))
        logprint("*** ninety fifth percentile relative diff between CORRECTED_DATA and {} is {} dB".format(refcolname, diff95))
        if diff95 > ninetyfifth_tolerance:
            raise RuntimeError("{}: diff {} dB exceeds tolerance of {} dB".format(cmd, diff95, ninetyfifth_tolerance))

d147_test_list = [
    ("GSOL_DATA", dict()),
    ("GSOL_DATA", dict(dist_ncpu=1)),
    ("GBSOL_DATA", dict(sol_jones="G,B", g_time_int=1, g_freq_int=0, b_time_int=0, b_freq_int=1)),
    ("PO_DATA", dict(g_type='phase-diag')),
    ("FS_DATA", dict(g_type='f-slope', g_time_int=1, g_freq_int=0)),
    ("TS_DATA", dict(g_type='t-slope', g_time_int=0, g_freq_int=1)),
    ("TFP_DATA", dict(g_type='tf-plane', g_time_int=0, g_freq_int=0)),
    ("DE_DATA", dict(model_list=os.path.abspath(os.path.join(basedir, '3C147-dE-apparent.lsm.html@dE')),
        sol_jones="G,dE", g_time_int=1, g_freq_int=1, de_time_int=60, de_freq_int=32))
]

d147_test_dict = dict(d147_test_list)

DEFAULT_REF_MS = "SUBSET-D147.MS"
DEFAULT_MS = "SUBSET-D147-output.MS"
DEFAULT_PARSET = "d147-test.parset"
DEFAULT_OUTPUT_DIR = os.environ["HOME"]+"/tmp"
DEFAULT_NCPU = None

def _ensure_output_ms(ms):
    logprint("*** output MS is {}".format(ms))
    if not os.path.exists(ms):
        tarball = ms + ".tgz"
        logprint("*** MS {} does not exist, will look for tarball {}".format(ms, tarball))
        if os.path.exists(tarball):
            os.chdir(os.path.dirname(tarball))
            cmd = "tar zxvf " + os.path.basename(tarball)
            logprint("*** running '{}' in {}".format(cmd, os.path.dirname(tarball)))
            if os.system(cmd):
                logprint("*** tarball {} failed to xtract".format(tarball))
                sys.exit(1)
        else:
            logprint("*** tarball {} doesn't exist either".format(tarball))
            sys.exit(1)


def d147_test(ms=DEFAULT_MS, refms=DEFAULT_REF_MS, parset=DEFAULT_PARSET, workdir=None, args=[], tests=d147_test_list):
    if "/" not in ms:
        _ensure_output_ms(os.path.join(basedir, ms))

    tester = SolverVerification(ms, refms, parset, workdir or DEFAULT_OUTPUT_DIR)
    for colname, opts in tests:
        if DEFAULT_NCPU:
            opts["dist_ncpu"] = DEFAULT_NCPU
        tester.verify(colname, args=args, **opts)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Runs cubical D147 tests')
    parser.add_argument('test', type=str, nargs="?", help='test to be run. Default is to run all.')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Optional extra arguments to gocubical.')
    parser.add_argument('-l', '--list', action='store_true', help='list all tests and exit')
    parser.add_argument('--ms', type=str, default=DEFAULT_MS, help='MS name')
    parser.add_argument('--refms', type=str, default=DEFAULT_REF_MS, help='reference MS name')
    parser.add_argument('--parset', type=str, default=DEFAULT_PARSET, help='base parset for tests')
    parser.add_argument('--genref', action='store_true', help='generate reference data. Default is to run tests')
    parser.add_argument('--dir', type=str, default=os.environ['HOME']+"/tmp", help='Directory for output')
    parser.add_argument('-j', '--ncpu', type=int, default=None, help='Number of cores to use')

    args = parser.parse_args()

    if args.list:
        logprint("Available tests: {}".format(" ".join([x[0] for x in d147_test_list])))
        sys.exit(0)

    DEFAULT_NCPU = args.ncpu

    if args.test is not None:
        tests = [ (args.test, d147_test_dict[args.test]) ]
    else:
        tests = d147_test_list

    _ensure_output_ms(args.ms)

    if args.genref:
        tester = SolverVerification(args.ms, args.refms, args.parset, args.dir)
        for colname, opts in tests:
            if DEFAULT_NCPU:
                opts["dist_ncpu"] = DEFAULT_NCPU
            tester.generate_reference(colname, args.args, **opts)
    else:
        d147_test(args.ms, args.refms, args.parset, args.dir, args=args.args, tests=tests)
