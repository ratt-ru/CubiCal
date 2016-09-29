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


