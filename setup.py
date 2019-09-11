try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils import setup, find_packages, Extension
    
from Cython.Build import build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = [ Extension("mogp_emulator.linalg.pivot_lapack", ["mogp_emulator/linalg/pivot_lapack.pyx"])]

setup(name='mogp_emulator',
      version='0.1',
      description='Tool for Multi-Output Gaussian Process Emulators',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='TBD',
      author='Eric Daub',
      author_email='edaub@turing.ac.uk',
      packages=find_packages(),
      license=['MIT'],
      install_requires=['numpy', 'scipy'],
      cmdclass = {'build_ext': build_ext}, 
      ext_modules = ext_modules)