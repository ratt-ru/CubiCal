Performance tuning
------------------

One of CubiCal's main features is high-speed gain calibration. However, its performance
is highly depenedent on its input parameters. This is unfortunate, but unavoidable given
variety of problems which CubiCal can solve. 

A very simple performance consideration for multi-core architectures is the number of processes 
which CubiCal spawns. This is specified by --dist-ncpu from command line or in the [dist] section 
of a parset. If this number is greater than one, multiprocessing will be used and one process will
dedicated to simulation/IO. Practially, the means that there is always one fewer process performing
compute than is specified. This becomes important when tiling the problem, as ideally the number of 
chunks per tile will be divisible by the number of processes minus one.

The second, and probably most crucial, step in making CubiCal as fast as possible is in the 
selection of time and frequency chunk sizes. If these chunks are too large, it will lead to a 
massive increase in cache-misses on the CPU. This will degrade performance quite substantially.
There is no hard-and-fast rule for selecting the sizes, but users should be aware of the negative
impact of setting them too large, as well as the wasted compute if they are too small. Due to
the dependence of this problem on architecture, it may take users a while to get a feel for the 
optimal. Note that soltion intervals can only be as large as a chunk; for large time/frequency 
solution intervals there is no alternative but to accept the decreased performance. 