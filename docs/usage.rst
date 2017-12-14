Basic Usage
-----------

Once CubiCal has been successfully installed, it can be run from command line using:

.. code:: bash

	gocubical

Adding the -h argument will print the help which provides all the command line arguments. 

CubiCal can be run in one of two ways; either by specifiying all the necessary 
arguments via the command line or by specifying a parset file. A parset file 
can be populated with all the arguments required to run a specific calibration.

A basic parset file looks something like this:

.. code-block:: none

	[data]
	ms = D147-LO-NOIFS-NOPOL-4M5S.MS/
	time-chunk = 32
	freq-chunk = 32

	[model]
	lsm = skymodels/3C147-GdB-spw0+pybdsm+apparent.lsm.html 
	column = 

	[montblanc] 
	dtype = double
	feed-type = circular
	mem-budget = 4096

	[sol]
	jones = G

	[out]
	column = CUSTOM_DATA

	[j1]
	time-int = 8
	freq-int = 8

If the above parset was named basic.parset, it could be run by invoking:

.. code-block:: bash

	gocubical basic.parset

This simple example only uses a fraction of the available options - unspecified options are
populated from the defaults. Square bracketed values are section headers which correspond to 
the first part of the associated command line argument e.g. the ms value in the [data] 
section would be specified on the command line as:

.. code-block:: bash

	gocubical --data-ms D147-LO-NOIFS-NOPOL-4M5S.MS/

This relationship can be inverted to add options to the parset. Consider the following example: 

.. code-block:: bash

	gocubical --dist-ncpu 4

Adding this to basic.parset is as simple as adding the [dist] section (the first part of the
command line argument), and specifying ncpu. basic.parset would then look as follows:

.. code-block:: none

	[data]
	ms = D147-LO-NOIFS-NOPOL-4M5S.MS/
	time-chunk = 32
	freq-chunk = 32

	[model]
	lsm = skymodels/3C147-GdB-spw0+pybdsm+apparent.lsm.html 
	column = 

	[montblanc] 
	dtype = double
	feed-type = circular
	mem-budget = 4096

	[sol]
	jones = G

	[out]
	column = CUSTOM_DATA

	[j1]
	time-int = 8
	freq-int = 8

	[dist]
	ncpu = 4

Note that a parset can be combined with options specified on the command line - the 
command line options will take precedence, making it easy to experiment without 
having to create a new parset.