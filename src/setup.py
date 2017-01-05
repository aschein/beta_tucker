#!/usr/bin/env pythonv
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


# ext_modules = [['**/*.pyx'],
#     Extension("sample", ["sample.pyx"], include_dirs=['fatwalrus'])
#     ]

setup(
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include(), '../../fatwalrus'],
    ext_modules=cythonize('**/*.pyx', include_path=['../../fatwalrus'])
)


# #!/usr/bin/env python
# import os
# import numpy as np
# from distutils.core import setup
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize

# # os.environ["CC"] = "g++-5"
# # os.environ["CXX"] = "g++-5"

# setup(name='fatwalrus',
#       version='1.0',
#       description='Code for laying out on the beach all day with the homies.',
#       author='Aaron Schein',
#       packages=['fatwalrus'],
#       cmdclass={'build_ext': build_ext},
#       include_dirs=[np.get_include(),
#                     os.path.expanduser('~/anaconda/include/')],
#       ext_modules=cythonize(['fatwalrus/**/*.pyx']))
