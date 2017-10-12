Python API
----------

This section documents all the functions which make up CubiCal. Much of this information 
is superflous for users, but is included for the sake of completeness and anyone wishing
to incorporate pieces of CubiCal into their own code. Additionally, contributors writing
their own gain machines may find this a useful resource when attempting to ensure 
compatibility with CubiCal's solver routine.

Submodules
~~~~~~~~~~

These modules make up the bulk of the Python functionality of CubiCal. The majority of
these are dedicated to data handling and book-keeping rather than the actual gain 
computation.

.. toctree::

    cubical.MBTiggerSim
    cubical.statistics
    cubical.solver
    cubical.TiggerSourceProvider
    cubical.data_handler
    cubical.flagging
    cubical.main
    cubical.param_db
    cubical.plots

Subpackages
~~~~~~~~~~~

These subpackages contain the gain machines and Cython kernels which perform various 
forms of gain calibration.

.. toctree::

    cubical.machines
    cubical.kernels
    


