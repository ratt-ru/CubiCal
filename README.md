# CubeCal

## Requirements

* numpy >= 1.11.3
* cython >= 0.25.2
* futures >= 3.0.5
* python-casacore >= 2.1.2

## Instructions

* Create and activate a virtualenv as described below.
* Clone the repository.
* To install all requirements EXCLUDING montblanc:

```
   $ pip install -r CubeCal/requirements.txt
```

* Install with:

```
   $ pip install CubeCal
```

* CubeCal can be run from command line using: 

```
   $ gocubecal
```	 

If this doesn't work, add path/to/your/virtual/env/bin/gocubecal to your path.

* -h or --help will display the currently available options.

## Setting up a virtual environment for the dependencies

On Ubuntu 14, make sure ``libcasacore2-dev`` is installed (not casacore21!). Then:

```
$ virtualenv cubecal
$ source cubecal/bin/activate
$ pip install -U numpy cython python-casacore
```




