"""
A custom command for distutils to facilitate testing of Asim
"""

from distutils.cmd import Command
from distutils.errors import DistutilsExecError

from os.path import abspath, dirname, join
import sys
import unittest


class test(Command):
    """Test Ceygen in the build directory"""

    description = 'run unit test-suite of Asim within the build directory'
    user_options = []

    def initialize_options(self):
        self.build_lib = None

    def finalize_options(self):
        self.set_undefined_options('build', ('build_lib', 'build_lib'))

    def run(self):
        self.run_command('build')  # build if not alredy run
        orig_path = sys.path[:]
        try:
            build_path = abspath(self.build_lib)
            sys.path.insert(0, build_path)
            import asim.tests as t
            assert dirname(t.__file__) == join(build_path, 'asim', 'tests')
            suite = unittest.TestLoader().loadTestsFromModule(t)
            result = unittest.TextTestRunner(verbosity=self.verbose).run(suite)
            if not result.wasSuccessful():
                raise Exception("There were test failures")
        #except Exception as e:
            #raise DistutilsExecError(e)
        finally:
            sys.path = orig_path
