from mathwrapper cimport fabs

cdef class Function:
    cdef double eval(self, double x) nogil:
        return 0.  # nothing better to return, cannot raise

cdef double find_root(Function func, Function derivative, double start, double precision,
                      int max_iters = 20) nogil except -1.:
    cdef double new, denom
    for i in range(max_iters):
        denom = derivative.eval(start)
        if denom == 0:
            with gil:
                raise ValueError("Division by zero, derivative({0}) is zero".format(start))
        new = start - func.eval(start)/denom
        if fabs(new - start) < precision:
            return new
        start = new
    with gil:
        raise ValueError("Cannot find root with precision {0} within {1} iterations".format(
                         precision, max_iters))
