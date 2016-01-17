#!/usr/bin/env python

# This file is part of pydpc.
#
# Copyright 2016 Christoph Wehmeyer
#
# pydpc is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, Extension
import versioneer

def extensions():
    from numpy import get_include
    from Cython.Build import cythonize
    ext_core = Extension(
        "pydpc.core",
        sources=["ext/core.pyx", "ext/_core.c"],
        include_dirs=[get_include()],
        extra_compile_args=["-O3", "-std=c99"])
    exts = [ext_core]
    return cythonize(exts)

class lazy_cythonize(list):
    """evaluates extension list lazyly.
    pattern taken from http://tinyurl.com/qb8478q"""
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

def long_description():
    ld = "Clustering by fast search and find of density peaks, designed by Alex Rodriguez"
    ld += " and Alessandro Laio, is a density-based clustering algorithm. The pydpc package"
    ld += " aims to make this algorithm available for Python users."
    return ld

setup(
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=lazy_cythonize(extensions),
    name='pydpc',
    version=versioneer.get_version(),
    description='A Python interface to Density Peak Clustering',
    long_description=long_description(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    keywords=[
        'cluster',
        'density'],
    url='https://github.com/cwehmeyer/pydpc',
    author='Christoph Wehmeyer',
    author_email='christoph.wehmeyer@fu-berlin.de',
    license='LGPLv3+',
    setup_requires=[
        'numpy>=1.7',
        'cython>=0.20',
        'setuptools>=0.6'],
    tests_require=[
        'numpy>=1.7',
        'nose>=1.3'],
    install_requires=[
        'numpy>=1.7',
        'matplotlib'],
    packages=['pydpc'],
    test_suite='nose.collector',
    scripts=[]
)

