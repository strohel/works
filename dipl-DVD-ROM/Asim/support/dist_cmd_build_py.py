from distutils.command.build_py import build_py as orig_build_py


class build_py(orig_build_py):
    """Little hack around distutils.command.build_py to extract a list of modules to
    cythonize out of py_modules and packages
    """

    def find_package_modules(self, package, package_dir):
        return self.filter_out_modules(orig_build_py.find_package_modules(self, package, package_dir))

    def find_modules(self):
        return self.filter_out_modules(orig_build_py.find_modules(self))

    def filter_out_modules(self, modules):
        ret = []
        for module in modules:
            # skip special modules:
            if module[1] == '__init__' or module[1] == '__main__':
                ret.append(module)
            else:
                self.distribution.modules_to_cythonize.add(module)
        return ret
