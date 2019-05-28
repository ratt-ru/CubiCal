*****************************
Requirements and Installation
*****************************

Ubuntu 18.04 
~~~~~~~~~~~~

CubiCal depends on python-casacore, the dependencies of which should be 
installed from the KERN-5 ppa. Note that ``apt-get install`` is ``apt install``
on 18.04.

.. code:: bash

	ENV DEB_DEPENDENCIES casacore-dev \
                     casacore-data \
                     build-essential \
                     python3-pip \ 
                     libboost-all-dev \ 
                     wcslib-dev \
                     git \
                     libcfitsio-dev
	apt-get update
	apt-get install -y $DEB_DEPENDENCIES
	pip3 install -U pip wheel setuptools
	python3.6 -m pip install -U .
	python3.6 -m pip install path/to/repo/

If you wish to install CubiCal in a virtual environment (recommended), see 
`Using a virtual environment`_. 

.. note:: 

	At this point, if CubiCal is required to predict model visiblities, it is necessary 
	to install Montblanc. The CPU version of montblanc is installed automatically if montblanc has not been previously
	installed. To install the GPU version of Montblanc, follow the instructions here_ before installing cubical.

	.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed from PyPI by running the following:

.. code:: bash

	pip install cubical

.. warning:: 

	To install in development mode, you will need to install some dependencies manually
	and cythonize the development kernels explicitly. Assuming that you have already
	cloned the repository, this can be done as follows:

	.. code:: bash

		pip install -e path/to/repo/

	The explicit cythonization step also allows for forced recythonization via ``--force`` or ``-f``:

	.. code:: bash

		python path/to/repo/setup.py gocythonize -f

Using a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing CubiCal in a virtual enviroment is highly recommended. To install
virtualenv using apt, run:

.. code:: bash

	apt-get install python3-virtualenv

To create a virtualenv, run:

.. code:: bash
	
	virtualenv -p python3 path/to/env/name

Activate the environment using:

.. code:: bash

	source path/to/env/name/bin/activate

This should change the command line prompt to be consistent with the virtualenv name.

It is often necessary to update pip, setuptools and wheel inside the environment:

.. code:: bash

	pip3 install -U pip setuptools wheel

Return to `Ubuntu 18.04 and 16.04`_ or `Ubuntu 14.04`_ to continue with installation.