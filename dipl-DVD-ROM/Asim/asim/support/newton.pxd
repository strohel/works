cdef class Function:
    cdef double eval(self, double x) nogil

cdef double find_root(Function func, Function derivative, double start, double precision,
                      int max_iters = *) nogil except -1.
