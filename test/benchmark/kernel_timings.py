import numpy as np
import timeit
import numpy.random
import cubical.kernels.cyfull_complex_omp as cyfull_omp
import cubical.kernels.cyfull_complex as cyfull
import cubical.kernels

def threads(n):
    """Sets the number of OMP threads to use"""
    cubical.kernels.num_omp_threads = n

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
    def __init__(self, nd=10, nm=1, nt=60, nf=32, na=28, dtype=np.complex128):
        self.o = np.zeros((nt,nf,na,na,2,2), dtype)
        self.m = np.zeros((nd,nm,nt,nf,na,na,2,2), dtype)
        self.r = np.zeros((nm,nt,nf,na,na,2,2), dtype)
        self.g = np.zeros((nd,nt,nf,na,2,2), dtype)
        self.f = np.zeros((nd,nt,nf,na),np.uint16)
        self.na = na
        self.baselines = [(p, q) for p in xrange(self.na) for q in xrange(self.na) if p < q]
        for p, q in self.baselines:
            fillrand(self.o[..., p, q, :, :])
            conj2x2(self.o[..., q, p, :, :], self.o[..., p, q, :, :])
            fillrand(self.m[..., p, q, :, :])
            conj2x2(self.m[..., q, p, :, :], self.m[..., p, q, :, :])
        for p in xrange(self.na):
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


def Time(code, name, n=3):
    res = timeit.repeat(code, repeat=n, number=1)
    print "{:70}: {:.2f}ms (best of {})".format(name, min(res)*1000, n)

if __name__ == "__main__":

    for nd in 1,10:
        u = UnorderedArrays(nd=nd,nt=60,na=24)
        o = OrderedArrays(u)
        print "\n### {} directions (model shape is {})\n".format(nd, u.m.shape)

        print('*** RES')

        Time(lambda:cyfull.cycompute_residual(u.m,u.g,u.gh,u.r,1,1), "compute_residual (DMTFAA order, native)")
        r0 = u.r.copy()

        o.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual_nomp(o.m,o.g,o.gh,o.r,1,1), "compute_residual new (no OMP, AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)

        o.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual_nomp_conj1(o.m,o.g,o.gh,o.r,1,1), "compute_residual new conj (no OMP, AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)

        o.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual_nomp_conj2(o.m,o.g,o.gh,o.r,1,1), "compute_residual new conj2 (no OMP, AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)

        o.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual(o.m,o.g,o.gh,o.r,1,1), "compute_residual new (OMP 1 thread, AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)

        threads(32)
        o.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual(o.m,o.g,o.gh,o.r,1,1), "compute_residual new (OMP 32 threads, AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)
        threads(1)

        u.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual_dmtfaa_xdir(u.m,u.g,u.gh,u.r,1,1), "compute_residual inner dir (DMTFAA order, native)")
        assert((u.r-r0).max()<1e-10)

        u.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual_dmtfaa_conj(u.m,u.g,u.gh,u.r,1,1), "compute_residual conj (DMTFAA order, native)")
        assert((u.r-r0).max()<1e-10)

        o.r.fill(0)
        Time(lambda:cyfull.cycompute_residual(o.m,o.g,o.gh,o.r,1,1), "compute_residual (*wrong* AAMTFD order, view)")
        assert((o.r-r0).max()<1e-10)

        u.r.fill(0)
        Time(lambda:cyfull_omp.cycompute_residual(u.m,u.g,u.gh,u.r,1,1), "compute_residual new (OMP, *wrong* DMTFAA order, native)")
        assert((o.r-r0).max()<1e-10)


        print('*** JH')

        Time(lambda:cyfull.cycompute_jh(u.m,u.g,u.jh,1,1), "compute_jh (DMTFAA order, native)")
        jh0 = u.jh.copy()

        o.jh.fill(0)
        Time(lambda:cyfull_omp.cycompute_jh_nomp(o.m,o.g,o.jh,1,1), "compute_jh new (no OMP, AAMTFD order, view)")
        assert((o.jh-jh0).max()<1e-10)

        o.jh.fill(0)
        Time(lambda:cyfull_omp.cycompute_jh(o.m,o.g,o.jh,1,1), "compute_jh new (OMP 1 thread, AAMTFD order, view)")
        assert((o.jh-jh0).max()<1e-10)

        o.jh.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_jh(o.m,o.g,o.jh,1,1), "compute_jh new (OMP 32 thread, AAMTFD order, view)")
        assert((o.jh-jh0).max()<1e-10)
        threads(1)

        print('*** JHR')

        u.r[:] = r0
        Time(lambda:cyfull.cycompute_jhr(u.jh,u.r,u.jhr,1,1), "compute_jhr (DMTFAA order, native)")
        jhr0 = u.jhr.copy()

        o.r[:] = r0
        o.jhr.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhr_nomp(o.jh,o.r,o.jhr,1,1), "compute_jhr new (no OMP, AAMTFD order, view)")
        assert((o.jhr-jhr0).max()<1e-10)

        o.r[:] = r0
        o.jhr.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhr(o.jh,o.r,o.jhr,1,1), "compute_jhr new (OMP 1 thread, AAMTFD order, view)")
        assert((o.jhr-jhr0).max()<1e-10)

        o.r[:] = r0
        o.jhr.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_jhr(o.jh,o.r,o.jhr,1,1), "compute_jhr new (OMP 32 thread, AAMTFD order, view)")
        assert((o.jhr-jhr0).max()<1e-10)
        threads(1)

        print('*** JHJ')

        u.jh[:] = jh0
        u.jhj.fill(0)
        Time(lambda:cyfull.cycompute_jhj(u.jh,u.jhj,1,1), "compute_jhj (DMTFAA order, native)")
        jhj0 = u.jhj.copy()

        o.jh[:] = jh0
        o.jhj.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhj_nomp(o.jh,o.jhj,1,1), "compute_jhj new (no OMP, AAMTFD order, view)")
        assert((o.jhj-jhj0).max()<1e-10)

        o.jh[:] = jh0
        o.jhj.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhj(o.jh,o.jhj,1,1), "compute_jhj new (OMP 1 thread, AAMTFD order, view)")
        assert((o.jhj-jhj0).max()<1e-10)

        o.jh[:] = jh0
        o.jhj.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_jhj(o.jh,o.jhj,1,1), "compute_jhj new (OMP 32 threads, AAMTFD order, view)")
        assert((o.jhj-jhj0).max()<1e-10)
        threads(1)

        print('*** JHJinv')

        u.jhj[:] = jhj0
        Time(lambda:cyfull.cycompute_jhjinv(u.jhj,u.jhjinv,u.f,1e-6,0), "compute_jhjinv (DMTFAA order, native)")
        jhjinv0 = u.jhjinv.copy()

        o.jhj[:] = jhj0
        o.jhjinv.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhjinv_nomp(o.jhj,o.jhjinv,o.f,1e-6,0), "compute_jhjinv new (no OMP, AAMTFD order, view)")
        assert((o.jhjinv-jhjinv0).max()<1e-10)

        o.jhj[:] = jhj0
        o.jhjinv.fill(0)
        Time(lambda:cyfull_omp.cycompute_jhjinv(o.jhj,o.jhjinv,o.f,1e-6,0), "compute_jhjinv new (OMP 1 thread, AAMTFD order, view)")
        assert((o.jhjinv-jhjinv0).max()<1e-10)

        o.jhj[:] = jhj0
        o.jhjinv.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_jhjinv(o.jhj,o.jhjinv,o.f,0,0), "compute_jhjinv new (OMP 32 threads, AAMTFD order, view)")
        assert((o.jhjinv-jhjinv0).max()<1e-10)
        threads(1)

        print('*** Update')

        Time(lambda:cyfull.cycompute_update(u.jhr,u.jhjinv,u.upd), "compute_update (DMTFAA order, native)")
        upd0 = u.upd.copy()

        o.upd.fill(0)
        Time(lambda:cyfull_omp.cycompute_update_nomp(o.jhr,o.jhjinv,o.upd), "compute_update new (no OMP, AAMTFD order, view)")
        assert((o.upd-upd0).max()<1e-10)

        o.upd.fill(0)
        Time(lambda:cyfull_omp.cycompute_update(o.jhr,o.jhjinv,o.upd), "compute_update new (OMP 1 thread, AAMTFD order, view)")
        assert((o.upd-upd0).max()<1e-10)

        o.upd.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_update(o.jhr,o.jhjinv,o.upd), "compute_update new (OMP 32 threads, AAMTFD order, view)")
        assert((o.upd-upd0).max()<1e-10)
        threads(1)

        print('*** Corrected')

        u.corr.fill(0)
        Time(lambda:cyfull.cycompute_corrected(u.o,u.g,u.gh,u.corr,1,1), "compute_corrected (DMTFAA order, native)")
        corr0 = u.corr.copy()

        o.corr.fill(0)
        Time(lambda:cyfull_omp.cycompute_corrected_nomp(o.o,o.g,o.gh,o.corr,1,1), "compute_corrected new (no OMP, AAMTFD order, view)")
        assert((o.corr-corr0).max()<1e-10)

        o.corr.fill(0)
        Time(lambda:cyfull_omp.cycompute_corrected_conj(o.o,o.g,o.gh,o.corr,1,1), "compute_corrected new conj (no OMP, AAMTFD order, view)")
        assert((o.corr-corr0).max()<1e-10)

        o.corr.fill(0)
        Time(lambda:cyfull_omp.cycompute_corrected(o.o,o.g,o.gh,o.corr,1,1), "compute_corrected new (OMP 1 thread, AAMTFD order, view)")
        assert((o.corr-corr0).max()<1e-10)

        o.corr.fill(0)
        threads(32)
        Time(lambda:cyfull_omp.cycompute_corrected(o.o,o.g,o.gh,o.corr,1,1), "compute_corrected new (OMP 32 threads, AAMTFD order, view)")
        assert((o.corr-corr0).max()<1e-10)
        threads(1)

        print('*** Apply ')

        mod = u.m.copy()
        Time(lambda:cyfull.cyapply_gains(mod,u.g,u.gh,1,1), "apply_gains (DMTFAA order, native)")
        mod0 = mod.copy()

        mod = o.m.copy()
        Time(lambda:cyfull.cyapply_gains(mod, o.g, o.gh, 1, 1), "apply_gains (*wrong* AAMTFD order, view)")
        assert((mod-mod0).max()<1e-10)

        mod = o.m.copy()
        Time(lambda:cyfull_omp.cyapply_gains_nomp(mod, o.g, o.gh, 1, 1), "apply_gains new (no OMP, AAMTFD order, view)")
        assert((mod-mod0).max()<1e-10)

        mod = o.m.copy()
        Time(lambda:cyfull_omp.cyapply_gains_conj(mod, o.g, o.gh, 1, 1), "apply_gains new conj (no OMP, AAMTFD order, view)")
        assert((mod-mod0).max()<1e-10)

        mod = o.m.copy()
        Time(lambda:cyfull_omp.cyapply_gains(mod, o.g, o.gh, 1, 1), "apply_gains new (OMP 1 thread, AAMTFD order, view)")
        assert((mod-mod0).max()<1e-10)

        mod = o.m.copy()
        threads(32)
        Time(lambda:cyfull_omp.cyapply_gains(mod, o.g, o.gh, 1, 1), "apply_gains new (OMP 32 threads, AAMTFD order, view)")
        assert((mod-mod0).max()<1e-10)
        threads(1)