import numpy as np
import timeit
import numpy.random
import sys
import pdb

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
    def __init__(self, nd=10, nm=1, nt=60, nf=32, na=28, t_int=1, f_int=1,
                 dtype=np.complex128,
                 ptype=np.complex128, pshape=None, jhrshape=None,
                 diaggain=False, diagmodel=False,
                 allocate=True,kernel=None):
        self._dtype = dtype
        self._ptype = ptype
        self.t_int, self.f_int = t_int, f_int
        if allocate:
            self.o = np.zeros((nt,nf,na,na,2,2), dtype)
            self.m = np.zeros((nd,nm,nt,nf,na,na,2,2), dtype)
            self.r = np.zeros((nm,nt,nf,na,na,2,2), dtype)
        # intervals?
        nt1 = nt/t_int + (1 if nt%t_int else 0)
        nf1 = nf/f_int + (1 if nf%f_int else 0)
        self._intshape = [nd,nt1,nf1,na,2,2]
        self._fullshape = [nd,nt,nf,na,2,2]
        self._paramgain = (pshape is not None)
        if self._paramgain:
            self._gshape = self._fullshape
            self._pshape = self._intshape
        else:
            self._gshape = self._pshape = self._intshape
        if allocate:
            self.f = np.zeros(self._pshape[:-2], np.uint16)
        if jhrshape:
            self._jhrshape = self._pshape[:-2] + jhrshape
        else:
            self._jhrshape = self._pshape
        if pshape:
            self._pshape = self._pshape[:-2] + pshape
        if allocate:
            self.g = np.zeros(self._gshape, dtype)
            self.p = np.zeros(self._pshape, ptype)
        self._kernel_name = refkern.__name__
        self.na = na
        self.baselines = [(p, q) for p in xrange(self.na) for q in xrange(self.na) if p < q]
        if allocate:
            for p, q in self.baselines:
                if diagmodel:
                    for c in 0,1:
                        fillrand(self.o[..., p, q, c, c])
                        fillrand(self.m[..., p, q, c, c])
                else:
                    fillrand(self.o[..., p, q, :, :])
                    fillrand(self.m[..., p, q, :, :])
                conj2x2(self.o[..., q, p, :, :], self.o[..., p, q, :, :])
                conj2x2(self.m[..., q, p, :, :], self.m[..., p, q, :, :])
            for p in xrange(self.na):
                if diaggain:
                    for c in 0,1:
                        fillrand(self.g[..., p, c, c])
                else:
                    fillrand(self.g[..., p, :, :])
            self.fillrest()

    def fillrest(self):
        self.gh = np.zeros_like(self.g)
        for p in xrange(self.na):
            conj2x2(self.gh[..., p, :, :], self.g[..., p, :, :])
        self.jh = np.zeros_like(self.m)
        self.jhr = np.zeros(self._jhrshape, self._ptype)
        self.jhj = np.zeros_like(self.jhr)
        self.jhjinv = np.zeros_like(self.jhj)
        self.corr = np.zeros_like(self.o)

    def printshapes(self):
        for name, value in self.__dict__.iteritems():
            if type(value) is np.ndarray:
                print("   .{}: {}".format(name, value.shape))

class OrderedArrays(UnorderedArrays):
    """Creates a set of test arrays for testing the cubical kernels. 
    These are reodered in a specific way, as given by the kernel argument"""
    def __init__(self, other, kernel, pshape=None, jhrshape=None):
        nd,nm,nt,nf,na = other.m.shape[:5]
        UnorderedArrays.__init__(self,nd,nm,nt,nf,na, other.t_int, other.f_int, other._dtype, other._ptype, pshape, jhrshape, allocate=False)
        # allocate arrays using preferred kernel ordering
        self.o = getattr(kernel,'allocate_vis_array')(other.o.shape, other._dtype)
        self.m = getattr(kernel,'allocate_vis_array')(other.m.shape, other._dtype)
        self.r = getattr(kernel,'allocate_vis_array')(other.r.shape, other._dtype)
        self.g = getattr(kernel,'allocate_gain_array')(other.g.shape, other._dtype)
        self.f = getattr(kernel,'allocate_gain_array')(other.f.shape, other.f.dtype)
        self.p = getattr(kernel,'allocate_param_array')(other.p.shape, other.p.dtype)
        # copy array content
        for arr in "omrgfp":
            np.copyto(getattr(self, arr), getattr(other, arr))
        # populate derived arrays
        self.fillrest()
        print "Array shapes are:"
        self.printshapes()


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

def benchmark_all(module, function_name, arguments, setup=None, check=None, notes=''):
    modname = module.__name__.split('.')[-1]
    # find all other variations of this function in the module
    for funcname in [function_name] + [n for n in dir(module) if n.startswith(function_name+"_") ]:
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
    parser.add_argument('--ti', type=int, default=1, help='time solution interval')
    parser.add_argument('--fi', type=int, default=1, help='freq solution interval')
    parser.add_argument('--nd', type=int, action='append', help='number of directions. Can use multiple times.')
    parser.add_argument('--diag', action='store_true', help='use diagonal gains')
    parser.add_argument('--diagmodel', action='store_true', help='use diagonal model')
    parser.add_argument('--omp', type=int, default=0, help='test OMP kernels with this number of threads')
    parser.add_argument('--pdb', action='store_true', help='drop into pdb on failures')

    parser.add_argument('--reference', type=str, default='cyfull_complex_reference', help='reference kernel')

    args = parser.parse_args()

    import cubical.kernels
    refkern_name = args.reference
    kernel_names = args.kernels or ['cyfull_complex']

    kernels = { name: cubical.kernels.import_kernel(name) for name in kernel_names }

    if 'cyfull_experimental' in kernel_names:
        cyfull_exp = kernels['cyfull_experimental']
    else:
        cyfull_exp = None

    refkern = cubical.kernels.import_kernel(refkern_name)
    testkerns = [ kernels[name] for name in kernel_names ]

    print "\n### Reference kernel:", refkern_name
    print "### Test kernels:"," ".join(kernel_names)

    nt, nf, na = args.nt, args.nf, args.na,

    t_int, f_int = args.ti, args.fi

    THREADS = [1] if not args.omp else [1, args.omp]
    NDIRS = [1] if not args.nd else args.nd

    print "### {} threads, {} dirs, {} times, {} freqs, {} antennas, intervals {} {}\n".format(THREADS,NDIRS,nt,nf,na,t_int,f_int)
    print "### ordered memory layout determined by {} kernel".format(kernel_names[0])

    def benchmark_function(function, arguments, setup=None, check=None):
        for kern in testkerns:
            if kern.__name__.endswith("omp"):
                for nthr in THREADS:
                    with threads(nthr):
                        benchmark_all(kern, function, arguments,
                                        setup=setup, check=check,
                                        notes="OMP {}T AAMTFD view".format(nthr))
            else:
                benchmark_all(kern, function, arguments,
                                setup=setup, check=check,
                                notes="AAMTFD view")


    for nd in NDIRS:
        ### testing for phase-slope kernels
        # python test/benchmark/kernel_timings.py cyf_slope cyf_slope_omp --reference cyf_slope_reference --omp 4 --diag --nd 10 --nf 128 --nt 10 --ti 5 --fi 16
        if "_slope" in refkern_name or "_plane" in refkern_name:

            ts = np.linspace(0, 1, nt)
            fs = np.linspace(0, 1, nf)
            if "tf_plane" in refkern_name:
                JHJAXIS = (3,4,1,5,2,0)       # scramble NEW JHJ axes into that order (w.r.t reference)
                JHRAXIS = (1,2,0,3,4,5)
                UPDAXIS = (1,2,0)
                nb, nparm = 6, 3
            elif "f_slope" in refkern_name or "t_slope" in refkern_name:
                JHJAXIS = (2,1,0)
                JHRAXIS = (1,0,2)
                UPDAXIS = (1,0)
            nb = len(JHJAXIS)
            nparm = len(UPDAXIS)
            dtype = np.complex128
            ftype = np.float64

            u = UnorderedArrays(nd=nd, nt=nt, nf=nf, na=na, t_int=t_int, f_int=f_int,
                                ptype=ftype, pshape=[nparm,2,2], jhrshape=[nb,2,2], diaggain=True,
                                diagmodel=args.diagmodel, kernel=refkern)
            o = OrderedArrays(u, testkerns[0], pshape=[nparm,2,2], jhrshape=[nb,2,2])

            print "\n### Testing {} directions, model shape is {}\n".format(nd, u.m.shape)

            print('*** RES')

            benchmark(lambda:refkern.cycompute_residual(u.m, u.g, u.gh, u.r, t_int, f_int), "{}.compute_residual (DMTFAA order, native)".format(refkern_name))
            r0 = u.r.copy()

            benchmark_function('cycompute_residual',(o.m, o.g, o.gh, o.r, t_int, f_int),
                            setup=lambda:o.r.fill(0), check=lambda:abs(o.r-r0).max()<1e-8)
            args.pdb and nfailed and pdb.set_trace()

            print('*** JH')

            benchmark(lambda:refkern.cycompute_jh(u.m, u.g, u.jh, t_int, f_int), "{}.compute_jh (DMTFAA order, native)".format(refkern_name))
            jh0 = u.jh.copy()

            benchmark_function('cycompute_jh', (o.m, o.g, o.jh, t_int, f_int),
                          setup=lambda: o.jh.fill(0), check=lambda: abs(o.jh - jh0).max() < 1e-8)
            args.pdb and nfailed and pdb.set_trace()

            print('*** JHR')
            jhr1shape = u._fullshape
            ujhr1 = np.zeros(jhr1shape, dtype)
            ojhr1 = testkerns[0].allocate_gain_array(jhr1shape, dtype, zeros=True)

            u.jhr.fill(0)
            benchmark(lambda: refkern.cycompute_tmp_jhr(u.gh, u.jh, u.r, ujhr1, 1, 1),
                      "{}.compute_tmp_jhr (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cycompute_inner_jhr', (o.gh, o.jh, o.r, ojhr1),
                               setup=lambda: ojhr1.fill(0),check=lambda:abs(ujhr1-ojhr1).max()<1e-9)

            benchmark(lambda: refkern.cycompute_jhr(ujhr1.real, u.jhr, ts, fs,  t_int, f_int),
                      "{}.compute_jhr (DMTFAA order, native)".format(refkern_name))

            # skipping J^H.R check as order of parameters has scrambled the blocks. Update should come right anyway.
            benchmark_function('cycompute_jhr', (ojhr1.real, o.jhr, ts, fs, t_int, f_int),
                               setup=lambda:o.jhr.fill(0),
                               check=lambda:abs(u.jhr-o.jhr[...,JHRAXIS,:,:]).max()<1e-6)

            args.pdb and nfailed and pdb.set_trace()

            print('*** JHJ')

            ujhj1shape = list(u.m.shape)
            del ujhj1shape[-3]
            ujhj1 = np.zeros(ujhj1shape, dtype)
            del ujhj1shape[1]  # omit model axis for new-style kernels (it was there in error)
            ojhj1 = np.zeros(ujhj1shape, dtype)

            benchmark(lambda: refkern.cycompute_tmp_jhj(u.m, ujhj1),"{}.compute_tmp_jhj (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cycompute_inner_jhj', (o.m, ojhj1),
                               setup=lambda:ojhj1.fill(0), check=lambda:abs(ujhj1[:,0,...]-ojhj1).max()<1e-9)

            u.jhj.fill(0)
            benchmark(lambda: refkern.cycompute_jhj(ujhj1.real, u.jhj, ts, fs, t_int, f_int),"{}.compute_jhj (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cycompute_jhj', (ojhj1.real, o.jhj, ts, fs, t_int, f_int),
                               setup=lambda:o.jhj.fill(0), check=lambda:abs(u.jhj-o.jhj[...,JHJAXIS,:,:]).max()<1e-6)

            args.pdb and nfailed and pdb.set_trace()

            print('*** JHJinv')

            benchmark(lambda: refkern.cycompute_jhjinv(u.jhj, u.jhjinv, 1e-6),
                      "{}.compute_jhjinv (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cycompute_jhjinv', (o.jhj, o.jhjinv, 1e-6),
                               check=lambda: abs(u.jhjinv - o.jhjinv[...,JHJAXIS,:,:]).max() < 1e-8)
            args.pdb and nfailed and pdb.set_trace()

            print('*** Update')
            benchmark(lambda: refkern.cycompute_update(u.jhr, u.jhjinv, u.p),
                      "{}.compute_update (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cycompute_update', (o.jhr, o.jhjinv, o.p),
                               check=lambda: abs(o.p[...,UPDAXIS,:,:] - u.p).max() < 1e-8)
            args.pdb and nfailed and pdb.set_trace()

            print('*** Construct gains')
            ug = np.zeros_like(u.g)
            og = np.zeros_like(o.g)

            benchmark(lambda: refkern.cyconstruct_gains(u.p, ug, ts, fs, t_int, f_int),
                      "{}.construct_gains (DMTFAA order, native)".format(refkern_name))

            benchmark_function('cyconstruct_gains', (o.p, og, ts, fs, t_int, f_int),
                               check=lambda: abs(og-ug).max() < 1e-8)
            args.pdb and nfailed and pdb.set_trace()

        ### testing for normal gain and phase-only kernels
        ### command lines e.g.
        #  python test/benchmark/kernel_timings.py cyphase_only cyphase_only_omp --reference cyphase_only_reference --omp 4 --diag --diagmodel --nd 10 --nf 128 --nt 10
        #  python test/benchmark/kernel_timings.py cyfull_complex cyfull_complex_omp cydiag_complex cydiagdiag_complex cydiagdiag_complex_omp --omp 4 --diag --diagmodel --nd 10 --nf 128 --nt 10
        else:
            u = UnorderedArrays(nd=nd, nt=nt, nf=nf, na=na, t_int=t_int, f_int=f_int, diaggain=args.diag,
                                diagmodel=args.diagmodel, kernel=refkern)
            o = OrderedArrays(u, testkerns[0])

            print "\n### Testing {} directions, model shape is {}\n".format(nd, u.m.shape)

            print('*** RES')

            benchmark(lambda:refkern.cycompute_residual(u.m, u.g, u.gh, u.r, t_int, f_int), "{}.compute_residual (DMTFAA order, native)".format(refkern_name))
            r0 = u.r.copy()

            benchmark_function('cycompute_residual',(o.m, o.g, o.gh, o.r, t_int, f_int),
                            setup=lambda:o.r.fill(0), check=lambda:abs(o.r-r0).max()<1e-8)
            args.pdb and nfailed and pdb.set_trace()

            # some one-off experimental tests, if the experimental kernel is being tested
            if cyfull_exp:
                u.r.fill(0)
                benchmark(lambda:cyfull_exp.cycompute_residual_dmtfaa_xdir(u.m, u.g, u.gh, u.r, t_int, f_int), "exp.compute_residual inner dir (DMTFAA order, native)")
                assert((u.r-r0).max()<1e-8)

                u.r.fill(0)
                benchmark(lambda:cyfull_exp.cycompute_residual_dmtfaa_conj(u.m, u.g, u.gh, u.r, t_int, f_int), "exp.compute_residual conj (DMTFAA order, native)")
                assert((u.r-r0).max()<1e-8)

                o.r.fill(0)
                benchmark(lambda:refkern.cycompute_residual(o.m, o.g, o.gh, o.r, t_int, f_int), "exp.compute_residual (*wrong* AAMTFD order, view)")
                assert((o.r-r0).max()<1e-8)

                u.r.fill(0)
                benchmark(lambda:cyfull_exp.cycompute_residual(u.m, u.g, u.gh, u.r, t_int, f_int), "exp.compute_residual new (OMP, *wrong* DMTFAA order, native)")
                assert((o.r-r0).max()<1e-8)


            print('*** JH')

            benchmark(lambda:refkern.cycompute_jh(u.m, u.g, u.jh, t_int, f_int), "{}.compute_jh (DMTFAA order, native)".format(refkern_name))
            jh0 = u.jh.copy()

            benchmark_function('cycompute_jh', (o.m, o.g, o.jh, t_int, f_int),
                          setup=lambda: o.jh.fill(0), check=lambda: abs(o.jh - jh0).max() < 1e-8)

            print('*** JHR')

            if "cyphase_only" in refkern_name:
                u.r[:] = r0
                benchmark(lambda: refkern.cycompute_jhr(u.gh, u.jh, u.r, u.jhr, t_int, f_int),
                          "{}.compute_jhr (DMTFAA order, native)".format(refkern_name))
                jhr0 = u.jhr.copy()

                benchmark_function('cycompute_jhr', (o.gh, o.jh, o.r, o.jhr, t_int, f_int),
                                   setup=lambda: (np.copyto(o.r, r0), np.copyto(o.jh, jh0), o.jhr.fill(0)),
                                   check=lambda: abs(o.jh - jh0).max() < 1e-8)
            else:
                u.r[:] = r0
                benchmark(lambda:refkern.cycompute_jhr(u.jh, u.r, u.jhr, t_int, f_int), "{}.compute_jhr (DMTFAA order, native)".format(refkern_name))
                jhr0 = u.jhr.copy()

                benchmark_function('cycompute_jhr', (o.jh, o.r, o.jhr, t_int, f_int),
                              setup=lambda: (np.copyto(o.r,r0),np.copyto(o.jh,jh0),o.jhr.fill(0)), check=lambda: abs(o.jh-jh0).max()<1e-8)

            args.pdb and nfailed and pdb.set_trace()

            print('*** JHJ')

            if "cyphase_only" in refkern_name:
                u.jh[:] = jh0
                u.jhj.fill(0)
                benchmark(lambda: refkern.cycompute_jhj(u.m, u.jhj, t_int, f_int),
                          "{}.compute_jhj (DMTFAA order, native)".format(refkern_name))
                jhj0 = u.jhj.copy()

                benchmark_function('cycompute_jhj', (o.m, o.jhj, t_int, f_int),
                                   setup=lambda: (np.copyto(o.jh, jh0), o.jhj.fill(0)),
                                   check=lambda: abs(o.jhj - jhj0).max() < 1e-8)
            else:
                u.jh[:] = jh0
                u.jhj.fill(0)
                benchmark(lambda:refkern.cycompute_jhj(u.jh, u.jhj, t_int, f_int), "{}.compute_jhj (DMTFAA order, native)".format(refkern_name))
                jhj0 = u.jhj.copy()

                benchmark_function('cycompute_jhj', (o.jh, o.jhj, t_int, f_int),
                              setup=lambda: (np.copyto(o.jh,jh0),o.jhj.fill(0)), check=lambda: abs(o.jhj-jhj0).max()<1e-8)

            args.pdb and nfailed and pdb.set_trace()

            print('*** JHJinv')

            u.jhj[:] = jhj0
            benchmark(lambda:refkern.cycompute_jhjinv(u.jhj, u.jhjinv, u.f, 1e-6, 0), "{}.compute_jhjinv (DMTFAA order, native)".format(refkern_name))
            jhjinv0 = u.jhjinv.copy()

            benchmark_function('cycompute_jhjinv', (o.jhj, o.jhjinv, o.f, 1e-6, 0),
                          setup=lambda: (np.copyto(o.jhj,jhj0),o.jhjinv.fill(0)), check=lambda: abs(o.jhjinv-jhjinv0).max()<1e-8)


            args.pdb and nfailed and pdb.set_trace()

            print('*** Update')

            if "cyphase_only" in refkern_name:
                uupd = np.zeros_like(u.p, u.o.real.dtype)
                oupd = np.zeros_like(o.p, o.o.real.dtype)
                benchmark(lambda:refkern.cycompute_update(u.jhr.real, u.jhjinv.real, uupd), "{}.compute_update (DMTFAA order, native)".format(refkern_name))
                upd0 = u.p.copy()

                benchmark_function('cycompute_update', (o.jhr.real, o.jhjinv.real, oupd),
                              setup=lambda: (np.copyto(o.jhr, jhr0), np.copyto(o.jhjinv, jhjinv0), o.p.fill(0)),
                              check=lambda: abs(oupd - uupd).max() < 1e-8)
            else:
                benchmark(lambda:refkern.cycompute_update(u.jhr, u.jhjinv, u.p), "{}.compute_update (DMTFAA order, native)".format(refkern_name))

                benchmark_function('cycompute_update', (o.jhr, o.jhjinv, o.p),
                              setup=lambda: (np.copyto(o.jhr, jhr0), np.copyto(o.jhjinv, jhjinv0), o.p.fill(0)),
                              check=lambda: abs(o.p - o.p).max() < 1e-8)

            args.pdb and nfailed and pdb.set_trace()

            if "cyphase_only" not in refkern_name:
                print('*** Corrected')

                u.corr.fill(0)
                benchmark(lambda:refkern.cycompute_corrected(u.o, u.g, u.gh, u.corr, t_int, f_int), "{}.compute_corrected (DMTFAA order, native)".format(refkern_name))
                corr0 = u.corr.copy()

                benchmark_function('cycompute_corrected', (o.o, o.g, o.gh, o.corr, t_int, f_int),
                              setup=lambda: o.corr.fill(0), check=lambda: abs(o.corr - corr0).max() < 1e-8)

                print('*** Apply ')

                mod = u.m.copy()
                benchmark(lambda:refkern.cyapply_gains(mod, u.g, u.gh, t_int, f_int), "{}.apply_gains (DMTFAA order, native)".format(refkern_name))
                mod0 = mod.copy()

                mod = o.m.copy()

                benchmark_function('cyapply_gains', (mod, o.g, o.gh, t_int, f_int),
                              setup=lambda: np.copyto(mod, o.m), check=lambda: abs(mod-mod0).max()<1e-8)

            args.pdb and nfailed and pdb.set_trace()


        assert(not nfailed)