import numpy as np
import timeit
import numpy.random
import sys

import cubical.kernels

def fillrand(x):
    """Fills a complex array with random values using numpy.random.rand"""
    x.real = numpy.random.rand(*x.shape)
    x.imag = numpy.random.rand(*x.shape)
    
def conj2x2(x,y):
    """Sets X=Y^H, where the last two axes of X and Y are [2,2]"""
    x[...,0,0] = y[...,0,0].conjugate()
    x[...,0,1] = y[...,1,0].conjugate()
    x[...,1,0] = y[...,0,1].conjugate()
    x[...,1,1] = y[...,1,1].conjugate()

def reroll_array(arr, axes):
    """Returns array where the axes are stored in a specific order"""
    axes = axes or range(len(arr.shape))
    realarray = arr.transpose(axes).copy(order='C')
    return realarray.transpose(np.argsort(axes))

class UnorderedArrays(object):
    """Creates a set of test arrays for testing the cubical kernels. These are in C order."""
    def __init__(self, nd=10, nm=1, nt=60, nf=32, na=28, dtype=np.complex128, diaggain=False, diagmodel=False):
        self.o = np.zeros((nt,nf,na,na,2,2), dtype)
        self.m = np.zeros((nd,nm,nt,nf,na,na,2,2), dtype)
        self.r = np.zeros((nm,nt,nf,na,na,2,2), dtype)
        self.g = np.zeros((nd,nt,nf,na,2,2), dtype)
        self.f = np.zeros((nd,nt,nf,na),np.uint16)
        self.na = na
        self.baselines = [(p, q) for p in xrange(self.na) for q in xrange(self.na) if p < q]
        for p, q in self.baselines:
            if diaggain:
                fillrand(self.o[..., p, q, (0,0), (1,1)])
                fillrand(self.m[..., p, q, (0,0), (1,1)])
            else:
                fillrand(self.o[..., p, q, :, :])
                fillrand(self.m[..., p, q, :, :])
            conj2x2(self.o[..., q, p, :, :], self.o[..., p, q, :, :])
            conj2x2(self.m[..., q, p, :, :], self.m[..., p, q, :, :])
        for p in xrange(self.na):
            if diaggain:
                fillrand(self.g[..., p, (0,0), (1,1)])
            else:
                fillrand(self.g[..., p, :, :])
        self.fillrest()

    def fillrest(self):
        self.gh = np.zeros_like(self.g)
        for p in xrange(self.na):
            conj2x2(self.gh[..., p, :, :], self.g[..., p, :, :])
        self.jh = np.zeros_like(self.m)
        self.jhr = np.zeros_like(self.g)
        self.jhj = np.zeros_like(self.g)
        self.jhjinv = np.zeros_like(self.g)
        self.upd = np.zeros_like(self.g)
        self.corr = np.zeros_like(self.o)


class OrderedArrays(UnorderedArrays):
    """Creates a set of test arrays for testing the cubical kernels. These are reodered in a specific way."""
    def __init__(self, other, data_axes=(2,3,0,1,4,5),
                       model_axes=(4,5,1,2,3,0,6,7),
                       res_axes=(3,4,0,1,2,5,6),
                       gain_axes=(3,1,2,0,4,5),
                       ):
        nd,nm,nt,nf,na = other.m.shape[:5]
        UnorderedArrays.__init__(self,nd,nm,nt,nf,na,other.m.dtype)
        self.o = reroll_array(other.o, data_axes)
        self.m = reroll_array(other.m, model_axes)
        self.r = reroll_array(other.r, res_axes)
        self.g = reroll_array(other.g, gain_axes)
        self.f = reroll_array(other.f, gain_axes[:-2])
        self.fillrest()


class threads():
    def __init__(self, nt):
        self.nt = nt
    def __enter__(self):
        cubical.kernels.num_omp_threads = self.nt
    def __exit__(self, type, value, traceback):
        cubical.kernels.num_omp_threads = 1

nfailed = 0

def benchmark(code, name, n=3):
    res = timeit.repeat(code, repeat=n, number=1)
    print "{:70}: {:.2f}ms (best of {})".format(name, min(res)*1000, n)

def benchmark_all(function, arguments, setup=None, check=None, notes=''):
    module, name = sys.modules[function.__module__], function.__name__
    modname = module.__name__.split('.')[-1]
    # find all other variations of this function in the module
    for funcname in [name] + [n for n in dir(module) if n.startswith(name+"_") ]:
        if setup is not None:
            setup()
        benchmark(lambda:getattr(module, funcname)(*arguments), "{}.{} ({})".format(modname, funcname, notes))
        if check is not None and not check():
            print "*** FAIL ***"
            global nfailed
            nfailed += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Runs cubical kernel timings')
    parser.add_argument('kernels', type=str, nargs='*', help='kernel(s) to be tested')
    parser.add_argument('--nt', type=int, default=60, help='number of timeslots')
    parser.add_argument('--nf', type=int, default=32, help='number of frequencies')
    parser.add_argument('--na', type=int, default=24, help='number of antennas')
    parser.add_argument('--nd', type=int, action='append', help='number of directions. Can use multiple times.')
    parser.add_argument('--diag', action='store_true', help='use diagonal gains')
    parser.add_argument('--diagmodel', action='store_true', help='use diagonal model')
    parser.add_argument('--omp', type=int, default=0, help='test OMP kernels with this number of threads')

    parser.add_argument('--reference', type=str, default='cyfull_complex_reference', help='reference kernel')

    args = parser.parse_args()

    import cubical.kernels
    kernel_names = [args.reference] + (args.kernels or ['cyfull_complex'])

    kernels = __import__('cubical.kernels', globals(), locals(), kernel_names, -1)

    if 'cyfull_experimental' in kernel_names:
        cyfull_exp = kernels.cyfull_experimental
    else:
        cyfull_exp = None

    refkern = getattr(kernels, kernel_names[0])
    testkerns = [ getattr(kernels, name) for name in kernel_names[1:] ]

    print "\n### Reference kernel:", kernel_names[0]
    print "### Test kernels:"," ".join(kernel_names[1:])

    nt, nf, na = args.nt, args.nf, args.na,
    THREADS = [1] if not args.omp else [1, args.omp]
    NDIRS = [1] if not args.nd else args.nd

    print "### {} threads, {} dirs, {} times, {} freqs, {} antennas\n".format(THREADS,NDIRS,nt,nf,na)

    def benchmark_function(function, arguments, setup=None, check=None):
        for kern in testkerns:
            if kern.__name__.endswith("omp"):
                for nthr in THREADS:
                    with threads(nthr):
                        benchmark_all(getattr(kern,function), arguments,
                                        setup=setup, check=check,
                                        notes="OMP {}T AAMTFD view".format(nthr))
            else:
                benchmark_all(getattr(kern,function), arguments,
                                setup=setup, check=check,
                                notes="AAMTFD view")


    for nd in NDIRS:
        u = UnorderedArrays(nd=nd,nt=nt,na=na,diaggain=args.diag, diagmodel=args.diagmodel)
        o = OrderedArrays(u)

        print "\n### Testing {} directions, model shape is {}\n".format(nd, u.m.shape)

        print('*** RES')

        benchmark(lambda:refkern.cycompute_residual(u.m, u.g, u.gh, u.r, 1, 1), "compute_residual (DMTFAA order, native)")
        r0 = u.r.copy()

        benchmark_function('cycompute_residual',(o.m, o.g, o.gh, o.r, 1, 1),
                        setup=lambda:o.r.fill(0), check=lambda:abs(o.r-r0).max()<1e-10)

        # some one-off experimental tests, if the experimental kernel is being tested
        if cyfull_exp:
            u.r.fill(0)
            benchmark(lambda:cyfull_exp.cycompute_residual_dmtfaa_xdir(u.m, u.g, u.gh, u.r, 1, 1), "compute_residual inner dir (DMTFAA order, native)")
            assert((u.r-r0).max()<1e-10)

            u.r.fill(0)
            benchmark(lambda:cyfull_exp.cycompute_residual_dmtfaa_conj(u.m, u.g, u.gh, u.r, 1, 1), "compute_residual conj (DMTFAA order, native)")
            assert((u.r-r0).max()<1e-10)

            o.r.fill(0)
            benchmark(lambda:refkern.cycompute_residual(o.m, o.g, o.gh, o.r, 1, 1), "compute_residual (*wrong* AAMTFD order, view)")
            assert((o.r-r0).max()<1e-10)

            u.r.fill(0)
            benchmark(lambda:cyfull_exp.cycompute_residual(u.m, u.g, u.gh, u.r, 1, 1), "compute_residual new (OMP, *wrong* DMTFAA order, native)")
            assert((o.r-r0).max()<1e-10)


        print('*** JH')

        benchmark(lambda:refkern.cycompute_jh(u.m, u.g, u.jh, 1, 1), "compute_jh (DMTFAA order, native)")
        jh0 = u.jh.copy()

        benchmark_function('cycompute_jh', (o.m, o.g, o.jh, 1, 1),
                      setup=lambda: o.jh.fill(0), check=lambda: abs(o.jh - jh0).max() < 1e-10)

        print('*** JHR')

        u.r[:] = r0
        benchmark(lambda:refkern.cycompute_jhr(u.jh, u.r, u.jhr, 1, 1), "compute_jhr (DMTFAA order, native)")
        jhr0 = u.jhr.copy()

        benchmark_function('cycompute_jhr', (o.jh, o.r, o.jhr, 1, 1),
                      setup=lambda: (np.copyto(o.r,r0),np.copyto(o.jh,jh0),o.jhr.fill(0)), check=lambda: abs(o.jh-jh0).max()<1e-10)


        print('*** JHJ')

        u.jh[:] = jh0
        u.jhj.fill(0)
        benchmark(lambda:refkern.cycompute_jhj(u.jh, u.jhj, 1, 1), "compute_jhj (DMTFAA order, native)")
        jhj0 = u.jhj.copy()

        benchmark_function('cycompute_jhj', (o.jh, o.jhj, 1, 1),
                      setup=lambda: (np.copyto(o.jh,jh0),o.jhj.fill(0)), check=lambda: abs(o.jhj-jhj0).max()<1e-10)

        print('*** JHJinv')

        u.jhj[:] = jhj0
        benchmark(lambda:refkern.cycompute_jhjinv(u.jhj, u.jhjinv, u.f, 1e-6, 0), "compute_jhjinv (DMTFAA order, native)")
        jhjinv0 = u.jhjinv.copy()

        benchmark_function('cycompute_jhjinv', (o.jhj, o.jhjinv, o.f, 1e-6, 0),
                      setup=lambda: (np.copyto(o.jhj,jhj0),o.jhjinv.fill(0)), check=lambda: abs(o.jhjinv-jhjinv0).max()<1e-10)


        print('*** Update')

        benchmark(lambda:refkern.cycompute_update(u.jhr, u.jhjinv, u.upd), "compute_update (DMTFAA order, native)")
        upd0 = u.upd.copy()

        benchmark_function('cycompute_update', (o.jhr, o.jhjinv, o.upd),
                      setup=lambda: (np.copyto(o.jhr, jhr0), np.copyto(o.jhjinv, jhjinv0), o.upd.fill(0)),
                      check=lambda: abs(o.upd - upd0).max() < 1e-10)

        print('*** Corrected')

        u.corr.fill(0)
        benchmark(lambda:refkern.cycompute_corrected(u.o, u.g, u.gh, u.corr, 1, 1), "compute_corrected (DMTFAA order, native)")
        corr0 = u.corr.copy()

        benchmark_function('cycompute_corrected', (o.o, o.g, o.gh, o.corr, 1, 1),
                      setup=lambda: o.corr.fill(0), check=lambda: abs(o.corr - corr0).max() < 1e-10)

        print('*** Apply ')

        mod = u.m.copy()
        benchmark(lambda:refkern.cyapply_gains(mod, u.g, u.gh, 1, 1), "apply_gains (DMTFAA order, native)")
        mod0 = mod.copy()

        mod = o.m.copy()

        benchmark_function('cyapply_gains', (mod, o.g, o.gh, 1, 1),
                      setup=lambda: np.copyto(mod, o.m), check=lambda: abs(mod-mod0).max()<1e-10)


        assert(not nfailed)