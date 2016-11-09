# CubeCal

## Requirements

* cython 0.23.4
* numpy 1.10.1
* python-casacore 2.1.2

## Instructions

The repository is confusing; there are more files than are strictly necessary.  
The important files at the moment are:

* full_complex.py
* cyfullms.py
* cyfull_complex.pyx
* setup_full_complex.py

In order to make use of the code, it is first necessary to run:  

python setup_full_complex.py build_ext --inplace -f  

in the code directory.

The --help provides information on the currently available options.

## Setting up a virtual environment for the dependencies

On Ubuntu 14, make sure ``libcasacore2-dev`` is installed (not casacore21!). Then:

```
$ virtualenv cubecal
$ source cubecal/bin/activate
$ pip install -U numpy cython python-casacore
```




