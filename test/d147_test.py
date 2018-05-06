#!/usr/bin/python

import os, os.path, sys
from casacore.tables import table
from collections import OrderedDict

def kw_to_args(**kw):
    return " ".join(["--{} {}".format(name.replace("_", "-"), value) for name, value in kw.items()])

basedir = os.path.dirname(__file__)

def logprint(arg):
    print>>sys.stderr,arg

class SolverVerification(object):
    def __init__(self, msname, parset, workdir="."):
        self.msname = os.path.abspath(msname if "/" in msname else os.path.join(basedir,msname))
        self.parset = os.path.abspath(parset if "/" in parset else os.path.join(basedir,parset))
        self.cmdline = "gocubical {} --data-ms {} ".format(self.parset, self.msname)
        os.chdir(workdir)
        logprint("*** Working directory is {}".format(os.getcwd()))

    def generate_reference(self, colname, args=[], **kw):
        cmd = self.cmdline + kw_to_args(out_column=colname, out_name="ref_"+colname, **kw)
        logprint("*** running {}".format(cmd))
        retcode = os.system(cmd)
        if retcode:
            raise RuntimeError("gocubical failed, return code {}".format(retcode))

    def verify(self, colname, args=[], tolerance=1e-6, **kw):
        cmd = self.cmdline + kw_to_args(out_column="CORRECTED_DATA", out_name="test_"+colname, **kw)
        logprint("*** running {}".format(cmd))
        retcode = os.system(cmd)
        if retcode:
            raise RuntimeError("gocubical failed, return code {}".format(retcode))
        tab = table(self.msname)
        cd = tab.getcol("CORRECTED_DATA")
        c0 = tab.getcol(colname)
        diff = abs(cd-c0).max()
        logprint("*** max diff between CORRECTED_DATA and {} is {}".format(colname, diff))
        if diff > tolerance:
            raise RuntimeError("diff {} exceeds tolerance of {}".format(diff, tolerance))

d147_test_list = [
    ("GSOL_DATA", dict()),
    ("GBSOL_DATA", dict(sol_jones="G,B", g_time_int=1, g_freq_int=0, b_time_int=0, b_freq_int=1))
]

d147_test_dict = dict(d147_test_list)

DEFAULT_MS = "SUBSET-D147.MS"
DEFAULT_PARSET = "d147-test.parset"
DEFAULT_OUTPUT_DIR = os.environ["HOME"]+"/tmp"
DEFAULT_NCPU = None

def d147_test(ms=DEFAULT_MS, parset=DEFAULT_PARSET, workdir=None, args=[]):
    os.system("pwd")
    tester = SolverVerification(ms, parset, workdir or DEFAULT_OUTPUT_DIR)
    for colname, opts in d147_test_list:
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

    if args.genref:
        tester = SolverVerification(args.ms, args.parset, args.dir)
        for colname, opts in d147_test_list:
            if args.tests and colname in args.tests:
                if DEFAULT_NCPU:
                    opts["dist_ncpu"] = DEFAULT_NCPU
                tester.generate_reference(colname, args.args, **opts)
    else:
        d147_test(args.ms, args.parset, args.dir, args=args.args)
