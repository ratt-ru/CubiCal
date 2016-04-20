from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_jhr(double complex [:,:,:,:,:,:] obser_arr,
                double complex [:,:,:,:,:,:] model_arr,
                double complex [:,:,:,:,:] gains):

    cdef int i,j,k,l,n = 0

    # cdef int [:,:] spec_eye = np.zeros([2,4], dtype=np.int32)

    cdef long new_shape[6]

    cdef double complex * o
    cdef double complex m

    cdef np.ndarray [np.complex128_t, ndim=6] RG
    # cdef np.ndarray[np.complex128_t, ndim=5] GHI, RGMH, out

    # spec_eye[0,0] = 1
    # spec_eye[1,3] = 1

    new_shape = model_arr.shape
    # new_shape = [1,1,14,4,1]

    with nogil, parallel(num_threads=4):
        for i in prange(new_shape[0], schedule="guided"):
            for j in xrange(new_shape[1]):
                for k in xrange(new_shape[2]):
                    for l in xrange(new_shape[3]):

                        model_arr[i,j,k,l,0,0] = (obser_arr[i,j,k,l,0,0]*
                                           gains[i,j,k,0,0]) + \
                                          (obser_arr[i,j,k,l,0,1]*
                                            gains[i,j,k,1,0])

                        model_arr[i,j,k,l,0,1] = (obser_arr[i,j,k,l,0,0]*
                                           gains[i,j,k,0,1]) + \
                                          (obser_arr[i,j,k,l,0,1]*
                                            gains[i,j,k,1,1])

                        model_arr[i,j,k,l,1,0] = (obser_arr[i,j,k,l,1,0]*
                                           gains[i,j,k,0,0]) + \
                                          (obser_arr[i,j,k,l,1,1]*
                                            gains[i,j,k,1,0])

                        model_arr[i,j,k,l,1,1] = (obser_arr[i,j,k,l,1,0]*
                                           gains[i,j,k,0,1]) + \
                                          (obser_arr[i,j,k,l,1,1]*
                                            gains[i,j,k,1,1])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void multiply(double complex *a, double complex *b, double complex *c,
                    int N, int K) nogil:
    cdef int j, k
    for j in range(2):
        for k in range(2):
            c[k] += a[j]*b[k+j*K]

    # model_arr = np.einsum("gh...ij,gh...jk->gh...ik", obser_arr, gains)

            # np.einsum("...ij,...kj->...ik", RG, model_arr[i].conj(), out=RG)
            #
            # RGMH = np.sum(RG, axis=-3)
            #
            # GHI = np.einsum("...ij,...jk->...ik", gains[i].conj(), spec_eye)
            #
            # RGMH = RGMH.reshape(new_shape)
            #
            # out = np.einsum("...ij,...jk->...ik", GHI, RGMH.reshape(
            #     new_shape))

    # return -2 * out.imag
