from complex_gain_kernel import *
import generics 

# Map the J^H.J inversion method to a generic inversion.
compute_jhjinv = generics.compute_2x2_inverse

# Map inversion to generic 2x2 inverse.
invert_gains = generics.compute_2x2_inverse