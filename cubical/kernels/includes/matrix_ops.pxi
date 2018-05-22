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

cdef inline void add_w_mat_product(complex3264 * out, const complex3264 *a, const complex3264 *w, const complex3264 *b) nogil:
    """
    Adds a 2x2 matrix product: out += A.B.W
    """
    out[0] += (a[0]*b[0]*w[0] + a[1]*b[2]*w[0])
    out[1] += (a[0]*b[1]*w[0] + a[1]*b[3]*w[0])
    out[2] += (a[2]*b[0]*w[0] + a[3]*b[2]*w[0])
    out[3] += (a[2]*b[1]*w[0] + a[3]*b[3]*w[0])


cdef inline void weight_upd_product(complex3264 *out, const complex3264 *r, const complex3264 *c, const float v, const int npol) nogil:
    """
    Multiplies 1X4 vector with 4x4 maxtrix and 4x1 vector : out = r.conj().c.r
    """

    cdef complex3264 denom 
    
    denom =  r[0].conjugate()*c[0]*r[0] + r[1].conjugate()*c[4]*r[0] + r[2].conjugate()*c[8]*r[0] + r[3].conjugate()*c[12]*r[0] + \
           r[0].conjugate()*c[1]*r[1] + r[1].conjugate()*c[5]*r[1] + r[2].conjugate()*c[9]*r[1] + r[3].conjugate()*c[13]*r[1] + \
           r[0].conjugate()*c[2]*r[1] + r[1].conjugate()*c[6]*r[2] + r[2].conjugate()*c[10]*r[2] +r[3].conjugate()*c[14]*r[2] + \
           r[0].conjugate()*c[3]*r[3] + r[1].conjugate()*c[7]*r[3] + r[2].conjugate()*c[11]*r[3] + r[3].conjugate()*c[15]*r[3]

    out[0] = (v+2*npol)/(v + 2*denom)



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

