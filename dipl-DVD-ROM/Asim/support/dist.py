from distutils.dist import Distribution

from .dist_cmd_build_ext import build_ext
from .dist_cmd_build_py import build_py
from .dist_cmd_test import test


class AsimDistribution(Distribution):

    def __init__(self, attrs):
        self.cflags = None  # Default CFLAGS overridable by setup.cfg
        self.ldflags = None  # Default LDFLAGS overridable by setup.cfg
        self.ext_options = None
        Distribution.__init__(self, attrs)
        self.cmdclass['build_py'] = build_py
        self.cmdclass['build_ext'] = build_ext
        self.cmdclass['test'] = test
        self.modules_to_cythonize = set()

    def has_ext_modules(self):
        # override, because at the time this is called the ext_modules are not yet collected
        return True
