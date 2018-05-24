*****************************
Requirements and Installation
*****************************

Ubuntu 18.04 and 16.04
~~~~~~~~~~~~~~~~~~~~~~

CubiCal depends on python-casacore, the dependencies of which should be 
installed from the KERN-3 ppa. Note that ``apt-get install`` is ``apt install``
on 18.04.

.. code:: bash

	apt-get install software-properties-common
	apt-add-repository -s ppa:kernsuite/kern-3
	apt-add-repository multiverse
	apt-add-repository restricted
	apt-get update
	apt-get install -y casacore-dev libboost-python-dev libcfitsio3-dev wcslib-dev

If you wish to install CubiCal in a virtual environment (recommended), see 
`Using a virtual environment`_. 

.. note:: 

	At this point, if CubiCal is required to predict model visiblities, it is necessary 
	to install Montblanc. To install Montblanc, follow the instructions here_.

	.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed from PyPI by running the following:

.. code:: bash

	pip install cubical

.. warning:: 

	To install in development mode, you will need to install some dependencies manually
	and cythonize the development kernels explicitly. Assuming that you have already
	cloned the repository, this can be done as follows:

	.. code:: bash

		pip install cython numpy
		python path/to/repo/setup.py gocythonize
		pip install -e path/to/repo/

	The explicit cythonization step also allows for forced recythonization via ``--force`` or ``-f``:

	.. code:: bash

		python path/to/repo/setup.py gocythonize -f


Ubuntu 14.04
~~~~~~~~~~~~

CubiCal depends on python-casacore, the dependencies of which should be 
installed from the radio-astro ppa.

.. code:: bash

	apt-get install software-properties-common
	apt-add-repository -s ppa:radio-astro/main
	apt-add-repository multiverse
	apt-add-repository restricted
	apt-get update
	apt-get install -y libboost-python-dev libcfitsio3-dev wcslib-dev libcasacore2-dev

If you wish to install CubiCal in a virtual environment (recommended), see 
`Using a virtual environment`_. 

.. warning:: 

	A special requirement on 14.04 is the installation of a specific version of python-casacore
	(to match the version of casacore in radio-astro). To install this dependency run:

	.. code:: bash

		pip install python-casacore==2.1.2

.. note:: 

	At this point, if CubiCal is required to predict model visiblities, it is necessary 
	to install Montblanc. To install Montblanc, follow the instructions here_.

	.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed from PyPI by running the following:

.. code:: bash

	pip install cubical

.. warning:: 

	To install in development mode, you will need to install some dependencies manually
	and cythonize the development kernels explicitly. Assuming that you have already
	cloned the repository, this can be done as follows:

	.. code:: bash

		pip install cython numpy
		python path/to/repo/setup.py gocythonize
		pip install -e path/to/repo/

	The explicit cythonization step also allows for forced recythonization via ``--force`` or ``-f``:

	.. code:: bash

		python path/to/repo/setup.py gocythonize -f

Using a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing CubiCal in a virtual enviroment is highly recommended. To install
virtualenv using pip, run:

.. code:: bash

	pip install virtualenv

To create a virtualenv, run:

.. code:: bash
	
	virtualenv path/to/env/name

Activate the environment using:

.. code:: bash

	source path/to/env/name/bin/activate

This should change the command line prompt to be consistent with the virtualenv name.

It is often necessary to update pip, setuptools and wheel inside the environment:

.. code:: bash

	pip install -U pip setuptools wheel

Return to `Ubuntu 18.04 and 16.04`_ or `Ubuntu 14.04`_ to continue with installation.