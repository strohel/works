from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list

from distutils import log
from distutils.command.build_ext import build_ext as orig_build_ext
from distutils.extension import Extension
from glob import glob
import os


class build_ext(orig_build_ext):

    user_options = orig_build_ext.user_options + [
        ('cflags=', None, "specify extra CFLAGS to pass to C and C++ compiler"),
        ('ldflags=', None, "specify extra LDFLAGS to pass to linker"),
        ('annotate', None, "pass --annotate to Cython when building extensions"),
    ]

    boolean_options = orig_build_ext.boolean_options + ['annotate']

    def initialize_options(self):
        orig_build_ext.initialize_options(self)
        self.cflags = None
        self.ldflags = None
        self.annotate = None

    def finalize_options(self):
        orig_build_ext.finalize_options(self)

        if self.cflags is None:
            self.cflags = self.distribution.cflags or []
        if isinstance(self.cflags, str):
            self.cflags = self.cflags.split()

        if self.ldflags is None:
            self.ldflags = self.distribution.ldflags or []
        if isinstance(self.ldflags, str):
            self.ldflags = self.ldflags.split()

    def run(self):
        modules = self.get_ext_modules()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []
        ext_options = self.distribution.ext_options or {}
        self.distribution.ext_modules.extend(cythonize(modules, annotate=self.annotate,
                force=self.force, build_dir=self.build_temp, **ext_options))

        self.extensions = self.distribution.ext_modules  # orig_build_ext caches the list
        orig_build_ext.run(self)

    def build_extension(self, ext):
        """HACK to actually apply cflags, ldflags"""
        orig_compile_args = ext.extra_compile_args
        ext.extra_compile_args = orig_compile_args or []
        ext.extra_compile_args.extend(self.cflags)
        orig_link_args = ext.extra_link_args
        ext.extra_link_args = orig_link_args or []
        ext.extra_link_args.extend(self.ldflags)

        orig_build_ext.build_extension(self, ext)

        ext.extra_compile_args = orig_compile_args
        ext.extra_link_args = orig_link_args

    def get_ext_modules(self):
        modules = []
        seen_modules = set()
        # we need to check all pure cython before Python ones, set doesn't preserve order,
        # therefore we use 2 nested iterations
        for module_set in (self.find_pure_cython_modules(), self.distribution.modules_to_cythonize):
            for module in module_set:
                module_name = module[0] + '.' + module[1]

                # skip later-found modules. That way we ignore foo.py if foo.pyx exists
                if module_name in seen_modules:
                    log.debug("skipping '{0}' ({1} module already seen)".format(module[2], module_name))
                    continue
                seen_modules.add(module_name)

                # because of the way cythonize works, language needs to be set before call to it
                ext_options = {}
                if self.distribution.ext_options and 'language' in self.distribution.ext_options:
                    ext_options['language'] = self.distribution.ext_options['language']
                modules.append(Extension(name=module_name, sources=[module[2]], **ext_options))
        return modules

    def find_pure_cython_modules(self):
        modules = set()
        build = self.distribution.get_command_obj('build_py')
        build.ensure_finalized()

        py_modules = self.distribution.py_modules or []
        for module in py_modules:
            # this is righly copied from build.find_modules():
            path = module.split('.')
            package = '.'.join(path[0:-1])
            module_base = path[-1]
            package_dir = build.get_package_dir(package)

            module_file = os.path.join(package_dir, module_base + ".pyx")
            if not os.path.isfile(module_file):
                continue
            modules.add((package, module_base, module_file))

        packages = self.distribution.packages or []
        for package in packages:
            # this is rougly copied from build.find_package_modules():
            package_dir = build.get_package_dir(package)
            module_files = glob(os.path.join(package_dir, "*.pyx"))
            for f in module_files:
                module = os.path.splitext(os.path.basename(f))[0]
                modules.add((package, module, f))
        return modules
