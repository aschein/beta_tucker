# Beta Tucker decomposition
Source code for the paper: Beta Tucker decomposition for DNA Methylation data by Aaron Schein, Pat Flaherty, Mingyuan Zhou, Dan Sheldon and Hanna Wallach, presented as a talk at the NIPS 2016 Workshop on Computational Biology.

The MIT License (MIT)

Copyright (c) 2016 Aaron Schein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What's included:
* [btd.pyx](https://github.com/aschein/btd/blob/master/src/btd.pyx): The main code file.  Implements Gibbs sampling inference for Beta Tucker decomposition.
* [mcmc_model.pyx](https://github.com/aschein/btd/blob/master/src/mcmc_model.pyx): Implements Cython interface for MCMC models.  Inherited by pgds.pyx.
* [sample.pyx](https://github.com/aschein/btd/blob/master/src/sample.pyx): Implements fast Cython method for sampling various distributions.
* [bessel.pyx](https://github.com/aschein/btd/blob/master/src/bessel.pyx): Implements fast Cython methods for sampling from the Bessel distribution.
* [Makefile](https://github.com/aschein/btd/blob/master/src/Makefile): Makefile (cd into this directoy and type 'make' to compile).
* [example.ipynb](https://github.com/aschein/btd/blob/master/src/example.ipynb): Jupyter notebook with examples of how to use the code.

## Dependencies:
* numpy
* matplotlib
* seaborn
* argparse
* path
* scikit-learn
* cython
* GSL
