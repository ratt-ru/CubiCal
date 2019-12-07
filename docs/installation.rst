*****************************
Requirements and Installation
*****************************

Ubuntu 18.04
~~~~~~~~~~~~

CubiCal depends on python-casacore, the dependencies of which should be
installed from the KERN-5_ ppa. The ppa can be added as follows:

.. _KERN-5: https://kernsuite.info/installation/

.. code:: bash

	sudo apt install software-properties-common
	sudo add-apt-repository -s ppa:kernsuite/kern-5
	sudo apt-add-repository multiverse
	sudo apt-add-repository restricted
	sudo apt update

Once the ppa has been added, CubiCal's dependencies can be installed as
follows:

.. code:: bash

	CUBICAL_DEPENDENCIES=(casacore-dev \
                     	      casacore-data \
                              build-essential \
                              python3-pip \
                              libboost-all-dev \
                              wcslib-dev \
                              git \
                              libcfitsio-dev)
	sudo apt install -y $CUBICAL_DEPENDENCIES

.. warning::

	A known casacore bug can cause data corruption on large reads. If your
	use-case falls in this category, please `build and install casacore from
	source`_.

	.. _`build and install casacore from source`: https://github.com/casacore/casacore

If you wish to install CubiCal in a virtual environment (recommended), see
`Using a virtual environment`_.

CubiCal can be installed by running the following:

.. code:: bash

	pip3 install git+https://github.com/ratt-ru/CubiCal.git@1.4.0

.. note::

	CubiCal can predict model visiblities using Montblanc_, but it is not installed
	by default. To install CubiCal with Montblanc, run:

	.. _Montblanc: https://montblanc.readthedocs.io

	.. code:: bash

		pip3 install "cubical[lsm-support]@git+https://github.com/ratt-ru/CubiCal.git@1.4.0"


.. warning::

	To install in development mode, assuming that you have already
	cloned the repository, run:

	.. code:: bash

		pip3 install -e path/to/repo/

	If you require Montblanc, run:

	.. code:: bash

		pip3 install -e path/to/repo/"[lsm-support]"


Using a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing CubiCal in a virtual enviroment is highly recommended. To install
virtualenv using apt, run:

.. code:: bash

	sudo apt install python3-virtualenv

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
