import sys, os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    print('path:', extName, extPath)
    return Extension(
        extName,
        [extPath],
        include_dirs = [".", np.get_include()],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-Wall"],
        extra_link_args = ['-g', '-Wno-cpp', '-ffast-math', '-O2'],
        define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )

# get the list of extensions
extNames = scandir("numflow")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

setup(
  name = 'numflow',         # How you named your package folder (MyLib)
  packages = ['numflow', 'numflow.cython'],   # Chose the same as "name"
  version = '0.0.5',          # Start with a small number and increase it with every change you make
  license='MIT',            # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Yet another visualization package',   # Give a short description about your library
  author = 'Vojtech Tomas',                   # Type in your name
  author_email = 'tomas@vojtatom.cz',      # Type in your E-Mail
  url = 'https://github.com/vojtatom/numflow',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/vojtatom/numflow/archive/0.0.5.tar.gz',    # I explain this later on
  keywords = ['visualization', 'data', 'flow'],   # Keywords that define your package best
  install_requires=[            # dependencies
    'Cython >= 0.18'
    ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  #packages = ["visual.modules.numeric", "visual.modules.numeric.data", "visual.modules.numeric.math"],
  ext_modules = cythonize(extensions),
  include_package_data=True
)