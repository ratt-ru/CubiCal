Requirements and Installation
-----------------------------

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

It is highly recommended to install CubiCal in a virtual enviroment. To install
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

It is necessary to update pip, setuptools and wheel inside the environment:

.. code:: bash

	pip install -U pip setuptools wheel

CubiCal also requires numpy to be installed. A special requirement on 14.04 is the
installation of a specific version of python-casacore. To install these dependencies 
run:

.. code:: bash

	pip install -U numpy python-casacore==2.1.2

At this point, if CubiCal is required to predict model visiblities, it is necessary 
to install Montblanc. To install Montblanc, follow the instructions here_.

.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed by running the following:

.. code:: bash

	pip install git+https://github.com/ratt-ru/CubiCal.git

To install in development mode, run the following instead:

.. code:: bash

	git clone https://github.com/ratt-ru/CubiCal.git
	pip install -e CubiCal/

Ubuntu 16.04
~~~~~~~~~~~~

CubiCal depends on python-casacore, the dependencies of which should be 
installed from the KERN-3 ppa.

.. code:: bash

	apt-get install software-properties-common
	apt-add-repository -s ppa:kernsuite/kern-3
	apt-add-repository multiverse
	apt-add-repository restricted
	apt-get update
	apt-get install -y casacore-dev libboost-python-dev libcfitsio3-dev wcslib-dev

It is highly recommended to install CubiCal in a virtual enviroment. To install
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

It is usually necessary to update pip, setuptools and wheel inside the environment:

.. code:: bash

	pip install -U pip setuptools wheel

CubiCal also requires numpy to be installed:

.. code:: bash

	pip install -U numpy

At this point, if CubiCal is required to predict model visiblities, it is necessary 
to install Montblanc. To install Montblanc, follow the instructions here_.

.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed by running the following:

.. code:: bash

	pip install git+https://github.com/ratt-ru/CubiCal.git

To install in development mode, run the following instead:

.. code:: bash

	git clone https://github.com/ratt-ru/CubiCal.git
	pip install -e CubiCal/
