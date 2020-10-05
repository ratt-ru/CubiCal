Example Parsets 
---------------

This page details available parset options. However, invoking ``gocubical -h``
should be preferred as it will always be up-to-date. The following is broken up
into the various sections of the parset.

.. note::

    The parsets provided in this section will work, but do not include every
    available option. For a full list, please see :ref:`Parset Options`.

Phase-only selfcal
##################

The following performs diagonal, phase-only selfcal using a measurement set 
column and produces corrected residuals.

.. code-block:: text

    [data]
    _Help = Visibility data options
    ms = ms.ms                              # Your measurement set name.
    column = DATA                           # Column in which the data lives.
    time-chunk = 100                        # Number of time slots in a chunk. 
                                            # Implicitly controls memory footprint.
    freq-chunk = 64                         # Number of frequencies in a chunk. 
                                            # Implicitly controls memory footprint.

    [sel]
    _Help = Data selection options
    field = 0                               # Select a field/fields. 
    ddid = None                             # Select spectral window/s.                   
    chan =                                  # Select channel/s.

    [out]
    _Help = Options for output products
    dir = cubical                           # Output directory name.
    name = cubical.cc-out/cubical           # Output path for non-visibility outputs.
    overwrite = True                        # Overwrite existing output.
    mode = sr                               # Produce corrected residuals.
                                            # Check --help for other options.
    column = CORRECTED_DATA                 # Output visibility column.
    
    [model]
    _Help = Calibration model options
    list = MODEL_DATA                       # Input visiblity column.
                                            # Can also be a Tigger .lsm file.
    
    [weight]
    _Help = Weighting options
    column = None                           # Weight column to use.

    [flags]
    _Help = General flagging options
    apply = -cubical                        # Ignore existing cubical flags.

    [sol]
    _Help = Solution options which apply at the solver level
    jones = G                               # Term/s to solve for.
                                            # Corresponding section/s below.
    delta-g = 1e-06                         # Stopping criteria on gains.
    delta-chi = 1e-06                       # Stopping criteria on chi-squared.
    term-iters = 20                         # Iteration recipe.
                                            # Overrides individual term settings.

    [dist]
    _Help = Parallelization and distribution options
    ncpu = 4                                # Number of processes to use.
                                            # Implicitly controls memory footprint.
    max-chunks = 12                         # Maximum number of simultaneous chunks.
                                            # Ideally >= ncpu.
    min-chunks = 12                         # Minimum number of simultaneous chunks.
                                            # Ideally = ncpu.
    safe = 1.0                              # Fraction of total memory CubiCal may use.
                                            # Will crash deliberately if exceeded.

    [g]
    _Help = Options for G-Jones term
    label = G                               # This term's name.
                                            # Must match [sol] section's jones option.
    type = phase-diag                       # Solve for a diagonal phase-only gain. 
    time-int = 1                            # Time interval for solution.
    freq-int = 1                            # Frequency interval for solution.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.

Phase and amplitude selfcal
############################

The following performs diagonal, phase and amplitude selfcal using a Tigger
.lsm sky model and produces corrected data. Note that using sky models requires 
CubiCal to be installed with lsm-support.

.. code-block:: text

    [data]
    _Help = Visibility data options
    ms = ms.ms                              # Your measurement set name.
    column = DATA                           # Column in which the data lives.
    time-chunk = 100                        # Number of time slots in a chunk. 
                                            # Implicitly controls memory footprint.
    freq-chunk = 64                         # Number of frequencies in a chunk. 
                                            # Implicitly controls memory footprint.

    [sel]
    _Help = Data selection options
    field = 0                               # Select a field/fields. 
    ddid = None                             # Select spectral window/s.                   
    chan =                                  # Select channel/s.

    [out]
    _Help = Options for output products
    dir = cubical                           # Output directory name.
    name = cubical.cc-out/cubical           # Output path for non-visibility outputs.
    overwrite = True                        # Overwrite existing output.
    mode = sc                               # Produce corrected data.
                                            # Check --help for other options.
    column = CORRECTED_DATA                 # Output visibility column.
    
    [model]
    _Help = Calibration model options
    list = skymodel.lsm.html                # Input sky model .lsm file.
                                            # Can also be a measurement set column.
    
    [weight]
    _Help = Weighting options
    column = None                           # Weight column to use.

    [flags]
    _Help = General flagging options
    apply = -cubical                        # Ignore existing cubical flags.

    [sol]
    _Help = Solution options which apply at the solver level
    jones = G                               # Term/s to solve for.
                                            # Corresponding section/s below.
    delta-g = 1e-06                         # Stopping criteria on gains.
    delta-chi = 1e-06                       # Stopping criteria on chi-squared.
    term-iters = 20                         # Iteration recipe.
                                            # Overrides individual term settings.

    [dist]
    _Help = Parallelization and distribution options
    ncpu = 4                                # Number of processes to use.
                                            # Implicitly controls memory footprint.
    max-chunks = 12                         # Maximum number of simultaneous chunks.
                                            # Ideally >= ncpu.
    min-chunks = 12                         # Minimum number of simultaneous chunks.
                                            # Ideally = ncpu.
    safe = 1.0                              # Fraction of total memory CubiCal may use.
                                            # Will crash deliberately if exceeded.

    [g]
    _Help = Options for G-Jones term
    label = G                               # This term's name.
                                            # Must match [sol] section's jones option.
    type = complex-diag                     # Solve for a diagonal complex gain. 
    time-int = 1                            # Time interval for solution.
    freq-int = 1                            # Frequency interval for solution.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.

Gain and bandpass selfcal
#########################

The following performs gain and bandpass calibration simultaneously,
using a measurement set column as input and produces uncorrected residuals.

.. code-block:: text

    [data]
    _Help = Visibility data options
    ms = ms.ms                              # Your measurement set name.
    column = DATA                           # Column in which the data lives.
    time-chunk = 100                        # Number of time slots in a chunk. 
                                            # Implicitly controls memory footprint.
    freq-chunk = 64                         # Number of frequencies in a chunk. 
                                            # Implicitly controls memory footprint.

    [sel]
    _Help = Data selection options
    field = 0                               # Select a field/fields. 
    ddid = None                             # Select spectral window/s.                   
    chan =                                  # Select channel/s.

    [out]
    _Help = Options for output products
    dir = cubical                           # Output directory name.
    name = cubical.cc-out/cubical           # Output path for non-visibility outputs.
    overwrite = True                        # Overwrite existing output.
    mode = ss                               # Produce uncorrected residuals.
                                            # Check --help for other options.
    column = CORRECTED_DATA                 # Output visibility column.
    
    [model]
    _Help = Calibration model options
    list = MODEL_DATA                       # Input visiblity column.
                                            # Can also be a Tigger .lsm file.
    
    [weight]
    _Help = Weighting options
    column = None                           # Weight column to use.

    [flags]
    _Help = General flagging options
    apply = -cubical                        # Ignore existing cubical flags.

    [sol]
    _Help = Solution options which apply at the solver level
    jones = B,G                              # Term/s to solve for.
                                            # Corresponding section/s below.
    delta-g = 1e-06                         # Stopping criteria on gains.
    delta-chi = 1e-06                       # Stopping criteria on chi-squared.
    term-iters = [20, 20, 20, 20]           # Iteration recipe. Loops over jones above. 
                                            # This will do 20 iterations on B,
                                            # 20 on G, 20 on B and finally 20 on G.
                                            # Overrides individual term settings.

    [dist]
    _Help = Parallelization and distribution options
    ncpu = 4                                # Number of processes to use.
                                            # Implicitly controls memory footprint.
    max-chunks = 12                         # Maximum number of simultaneous chunks.
                                            # Ideally >= ncpu.
    min-chunks = 12                         # Minimum number of simultaneous chunks.
                                            # Ideally = ncpu.
    safe = 1.0                              # Fraction of total memory CubiCal may use.
                                            # Will crash deliberately if exceeded.

    [g]
    _Help = Options for G-Jones term
    label = G                               # This term's name.
                                            # Must match [sol] section's jones option.
    type = complex-2x2                      # Solve for a full 2x2 complex gain. 
                                            # This can be restircted using update type.
    time-int = 0                            # Time interval for solution.
                                            # 0 is the entire chunk axis.
    freq-int = 1                            # Frequency interval for solution.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.

    [b]
    _Help = Options for G-Jones term
    label = b                               # This term's name.
                                            # Must match [sol] section's jones option.
    type = complex-2x2                      # Solve for a full 2x2 complex gain. 
                                            # This can be restircted using update type.
    time-int = 1                            # Time interval for solution.
    freq-int = 0                            # Frequency interval for solution.
                                            # 0 is the entire chunk axis.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.


Direction-independent and direction-dependent selfcal
#####################################################

The following performs DI and DD gain calibration simultaneously,
using a tagged sky model as input and produces corrected residuals.
Note that using sky models requires CubiCal to be installed with lsm-support.

.. code-block:: text

    [data]
    _Help = Visibility data options
    ms = ms.ms                              # Your measurement set name.
    column = DATA                           # Column in which the data lives.
    time-chunk = 100                        # Number of time slots in a chunk. 
                                            # Implicitly controls memory footprint.
    freq-chunk = 64                         # Number of frequencies in a chunk. 
                                            # Implicitly controls memory footprint.

    [sel]
    _Help = Data selection options
    field = 0                               # Select a field/fields. 
    ddid = None                             # Select spectral window/s.                   
    chan =                                  # Select channel/s.

    [out]
    _Help = Options for output products
    dir = cubical                           # Output directory name.
    name = cubical.cc-out/cubical           # Output path for non-visibility outputs.
    overwrite = True                        # Overwrite existing output.
    mode = sr                               # Produce DI corrected residuals.
                                            # Corrected data cannot be produced for DD gains. 
                                            # Check --help for other options.
    column = CORRECTED_DATA                 # Output visibility column.
    
    [model]
    _Help = Calibration model options
    list = skymodel.lsm.html+-skymodel.lsm.html@dE:skymodel.lsm.html@dE
                                            # Input recipe.
                                            # This creates a direction dependent model.
                                            # Directions are separated by colons.
                                            # In this example, direction 0 will be
                                            # the entire sky model minus the contributions
                                            # from the tagged dE sources. The remaining
                                            # directions will be those tagged in The
                                            # sky model. Multiple columns can be specified 
                                            # in a similar fashion.

    [weight]
    _Help = Weighting options
    column = None                           # Weight column to use.

    [flags]
    _Help = General flagging options
    apply = -cubical                        # Ignore existing cubical flags.

    [sol]
    _Help = Solution options which apply at the solver level
    jones = G,dE                           # Term/s to solve for.
                                            # Corresponding section/s below.
    delta-g = 1e-06                         # Stopping criteria on gains.
    delta-chi = 1e-06                       # Stopping criteria on chi-squared.
    term-iters = [20, 20, 20, 20]           # Iteration recipe. Loops over jones above. 
                                            # This will do 20 iterations on G,
                                            # 20 on dE, 20 on G and finally 20 on dE.
                                            # Overrides individual term settings.

    [dist]
    _Help = Parallelization and distribution options
    ncpu = 4                                # Number of processes to use.
                                            # Implicitly controls memory footprint.
    max-chunks = 12                         # Maximum number of simultaneous chunks.
                                            # Ideally >= ncpu.
    min-chunks = 12                         # Minimum number of simultaneous chunks.
                                            # Ideally = ncpu.
    safe = 1.0                              # Fraction of total memory CubiCal may use.
                                            # Will crash deliberately if exceeded.

    [g]
    _Help = Options for G-Jones term
    label = G                               # This term's name.
                                            # Must match [sol] section's jones option.
    type = complex-2x2                      # Solve for a full 2x2 complex gain. 
                                            # This can be restircted using update type.
    time-int = 1                            # Time interval for solution.
                                            # 0 is the entire chunk axis.
    freq-int = 1                            # Frequency interval for solution.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.

    [de]
    _Help = Options for G-Jones term
    label = dE                              # This term's name.
                                            # Must match [sol] section's jones option.
    type = complex-2x2                      # Solve for a full 2x2 complex gain. 
                                            # This can be restircted using update type.
    dd-term = 1                             # This term is diretion dependent.
    time-int = 20                           # Time interval for solution.
    freq-int = 32                           # Frequency interval for solution.
                                            # 0 is the entire chunk axis.
    max-iter = 20                           # Maximum number of iterations.
                                            # May be ignored if [sol] term-iters is set.
    _Templated = True                       # Other parameters polulated from defaults.
                                            # Can be safely ignored.