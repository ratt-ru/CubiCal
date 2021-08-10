from __future__ import print_function
from psutil import virtual_memory
from cubical import workers

from cubical.tools import logger, ModColor
log = logger.getLogger("wisdom")


def estimate_mem(data_handler, tile_list, data_opts, dist_opts):

    # Grab current memory information.

    mem = virtual_memory()

    tot_sys_mem = mem.total/(1024**3)  # Convert to value in GiB.

    # Grab an example tile - for now we assume that it is a representative
    # example. In principle we could loop over all tiles and look for the
    # absolute peak memeory usage. We do some fiddling to get the number of
    # chunks along each axis.

    example_tile = tile_list[0]
    chunks_per_tile = example_tile.total_tf_chunks()
    along_freq = max(len(cf) for cf in data_handler.freqchunks.values())
    along_time = chunks_per_tile//along_freq

    # Figure out the chunk dimensions. The ors handle the 0 case. TODO: The
    # time behaviour might not be safe.

    t_per_chunk = example_tile.rowchunks[0].timeslice.stop
    f_per_chunk = data_opts["freq-chunk"] or data_handler.all_freqs.size

    # Grab some necessary dimensions. n_dir may be zero when transferring.
    # The or catches this case.

    n_dir = len(data_handler.model_directions) or 1
    n_ant = data_handler.nants
    n_bl = (n_ant*(n_ant - 1))/2
    n_corr = example_tile.ncorr

    # Figure out the data points per tile. This is done assuming that the data
    # is still in rowlike format at this point. TODO: Verify behaviour with
    # multiple directions.

    data_points_per_tile = \
        n_dir*(along_time*t_per_chunk*n_bl)*(along_freq*f_per_chunk)*n_corr

    # Figure out the data points per chunk. This is done assuming that the
    # data is now going to be in the internal data representation. TODO:
    # verify that this works correctly when varying n_dir and n_ant.

    data_points_per_chunk = \
        n_dir*t_per_chunk*f_per_chunk*n_ant*n_ant*n_corr

    # Empirically determined values for the slope and intercept for each
    # solver process.

    w_slope, w_intercept = 5.77281225e-08, 4.53587065e-01

    w_mem_est = w_slope*data_points_per_chunk + w_intercept

    # Empirically determined values for the slope and intercept for the
    # I/O process.

    io_slope, io_intercept = 4.24736385e-08, 3.73958333e-01

    io_mem_est = io_slope*data_points_per_tile + io_intercept

    # Make a guess at the total memory use.

    tot_mem_est = w_mem_est*(workers.num_workers or 1) + io_mem_est

    # Log all of this to terminal.

    print("Detected a total of {:.2f}GiB of system memory.".format(
          tot_sys_mem), file=log)
    print("Per-solver (worker) memory use estimated at {:.2f}GiB: {:.2f}% of "
          "total system memory.".format(w_mem_est, 100*w_mem_est/tot_sys_mem),
          file=log)
    print("Peak I/O memory use estimated at {:.2f}GiB: {:.2f}% of total "
          "system memory.".format(io_mem_est, 100*io_mem_est/tot_sys_mem),
          file=log)
    print("Total peak memory usage estimated at {:.2f}GiB: {:.2f}% of total "
          "system memory.".format(tot_mem_est, 100*tot_mem_est/tot_sys_mem),
          file=log)

    if tot_mem_est > tot_sys_mem*dist_opts["safe"] and dist_opts["safe"]:

        raise MemoryError(
            "Estimated memory usage exceeds allowed pecentage of system "
            "memory. Memory usage can be reduced by lowering the number of "
            "chunks, the dimensions of each chunk or the number of worker "
            "processes. This error can suppressed by setting --dist-safe to "
            "zero.")

    return
