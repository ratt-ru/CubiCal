cdef inline void mat_product(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Computes a 2x2 matrix product: out = A.B
    A matrix is just a sequence in memory of four complex numbers [x00,x01,x10,x11] (so e.g. the last 2,2 axes of a cube)
    """
    out[0] = (a[0]*b[0] + a[1]*b[2])
    out[1] = (a[0]*b[1] + a[1]*b[3])
    out[2] = (a[2]*b[0] + a[3]*b[2])
    out[3] = (a[2]*b[1] + a[3]*b[3])

cdef inline void add_mat_product(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Adds a 2x2 matrix product: out += A.B
    """
    out[0] += (a[0]*b[0] + a[1]*b[2])
    out[1] += (a[0]*b[1] + a[1]*b[3])
    out[2] += (a[2]*b[0] + a[3]*b[2])
    out[3] += (a[2]*b[1] + a[3]*b[3])


cdef inline void mat_product_diag(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Computes a 2x2 matrix product of two diagonal matrices: out = A.B
    """
    out[0] = a[0]*b[0]
    out[1] = out[2] = 0
    out[3] = a[3]*b[3]

cdef inline void add_mat_product_diag(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Adds a 2x2 matrix product of two diagonal matrices: out += A.B
    """
    out[0] += a[0]*b[0]
    out[3] += a[3]*b[3]

cdef inline void mat_conjugate(complex3264 * out,const complex3264 *x) nogil:
    """
    Computes a 2x2 matrix Hermitian conjugate: out = X^H
    """
    out[0] = x[0].conjugate()
    out[1] = x[2].conjugate()
    out[2] = x[1].conjugate()
    out[3] = x[3].conjugate()

cdef inline void mat_conjugate_diag(complex3264 * out,const complex3264 *x) nogil:
    """
    Computes a 2x2 matrix Hermitian conjugate: out = X^H
    """
    out[0] = x[0].conjugate()
    out[1] = out[2] = 0
    out[3] = x[3].conjugate()

