Requirements and Installation
-----------------------------

Ubuntu 14.04
~~~~~~~~~~~~

A special requirement for installation on Ubuntu 14.04 is the installation of Python 
development headers. These can be installed by running:

.. code:: bash

	apt-get install python-dev

CubiCal depends on python-casacore which should be installed from the radio-astro ppa.

.. code:: bash

	apt-get install software-properties-common
	apt-add-repository -s ppa:radio-astro/main
	apt-add-repository multiverse
	apt-add-repository restricted
	apt-get update
	apt-get install -y python-casacore

It highly recommended to install CubiCal in a virtual enviroment. To install
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

For compatibility with Tigger sky models, it is necessary to install vext.pyqy4:

.. code:: bash

	pip install vext.pyqt4

At this point, if simultion mode is required, it is necessary to install Montblanc. 
To install Montblanc, follow the instructions here_.

.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed by running the following:

.. code:: bash

	git clone https://github.com/ratt-ru/CubiCal.git
	pip install -r CubiCal/requirements.txt
	pip install CubiCal/

To install in development mode, run the following instead:

.. code:: bash

	pip install -e CubiCal/

Ubuntu 16.04
~~~~~~~~~~~~

CubiCal depends on python-casacore which should be installed from the kern-2 ppa.

.. code:: bash

	apt-get install software-properties-common
	apt-add-repository -s ppa:kernsuite/kern-2
	apt-add-repository multiverse
	apt-add-repository restricted
	apt-get update
	apt-get install -y python-casacore

It highly recommended to install CubiCal in a virtual enviroment. To install
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

For compatibility with Tigger sky models, it is necessary to install vext.pyqy4:

.. code:: bash

	pip install vext.pyqt4

At this point, if simulation mode is required, it is necessary to install Montblanc. 
To install Montblanc, follow the instructions here_.

.. _here: https://montblanc.readthedocs.io

CubiCal can now be installed by running the following:

.. code:: bash

	git clone https://github.com/ratt-ru/CubiCal.git
	pip install -r CubiCal/requirements.txt
	pip install CubiCal/

To install in development mode, run the following instead:

.. code:: bash

	pip install -e CubiCal/
