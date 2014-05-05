#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from distutils.core import setup

from support.dist import AsimDistribution


cython_directives = {
    "profile": True,  # add profiling information to generated code, very small overhead
    "infer_types": True,  # type more things automatically
#    "boundscheck": False,  # don't check array indexing index
    "cdivision": True,  # C division semantics, don't check for zero division
    "binding": False,  # default was changed to True in Cython commit 621dbe6403 and it
                       # breaks the build. I don't know what it means, it's undocumented.
}
ext_options = {
    "compiler_directives": cython_directives,
#    "pyrex_gdb": True,  # enable if you want to use cygdb
}

setup(
    packages=['asim', 'asim.assimilation', 'asim.dispmodel', 'asim.simulation',
              'asim.support', 'asim.tests'],
    package_data={'asim.simulation': ['meteo.mat', 'receptory_ETE'],
                  'asim.support': ['teme2_full.png', 'teme_osm_aligned.png']},
    distclass=AsimDistribution,
    include_dirs=[np.get_include()],  # default overridable by setup.cfg
    ext_options=ext_options,
    cflags=['-O2', '-march=native'], #, '-fopenmp'],  # default overridable by setup.cfg
    #ldflags=['-fopenmp'],  # ditto

    # meta-data; see http://docs.python.org/distutils/setupscript.html#additional-meta-data
    name='asim',
    author='Matěj Laitl, Václav Šmídl, Radek Hofman',
    author_email='matej@laitl.cz',
)
