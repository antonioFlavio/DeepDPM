#!/usr/bin/env python
from setuptools import setup, find_packages
import distutils
from pip._internal.req import parse_requirements

try:
    distutils.dir_util.remove_tree('dist')
    distutils.dir_util.remove_tree('build')
except:
    print("distutils can't remove 'dist' and 'build'")

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements-new.txt', session=False)

# reqs is a list of requirement
try:
    dependencies = [str(ir.req) for ir in install_reqs]
except:
    dependencies = [str(ir.requirement) for ir in install_reqs]

VERSION = '0.1.1'

setup(name='deep-dpm-antoniofork',
      version=VERSION,
      author='git@antonioFlavio',
      install_requires=dependencies,
      # Package info
      packages=find_packages(),
      zip_safe=True,
      include_package_data=True
      ),
