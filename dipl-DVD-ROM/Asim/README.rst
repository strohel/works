============
Asim Project
============

Building & Installing
=====================

Current dependencies:
 * Ceygen 0.3 or newer
 * PyBayes, latest tip of the `experimental` branch
 * Cython 0.19.1 or newer

``python setup.py build`` to build. See ``python setup.py build_ext -h`` for possible
options for the important build_ext subcommand.

Type ``python setup.py test`` to run the simulation and assimilation in a twin
experiment.

Type ``python setup.py install`` to install.

Launching
=========

Simple simulation can be starting by calling:

``python -c 'import asim.simulation.simple as s; s.main()'``

Assimilation can be started using

``python -c 'import asim.assimilation.twin as t; t.main()'``

Calling something like `python simulation/simple.py` doesn't work for Cython build because
it circumvents binary version of the simulation.simple module. The main() methods may take
optional parameters, study the sources.
