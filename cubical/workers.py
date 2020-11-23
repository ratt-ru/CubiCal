from __future__ import print_function
from builtins import range
import multiprocessing, os, sys, traceback
import concurrent.futures as cf
import re
import numba


import cubical.kernels
from cubical.tools import logger
from cubical import solver


log = logger.getLogger("main")

worker_process_properties = dict(MainProcess={})

# number of worker processes that will be run
num_workers = 0

def _setup_workers_and_threads(force_serial, ncpu, nworkers, nthreads, montblanc_threads, max_workers):
    """
    Internal helper -- determines how many workers and threads to allocate, based on defaults specified. See discussion in
    https://github.com/ratt-ru/CubiCal/pull/171#issuecomment-388334586

    Returns:
        Tuple of parallel,nworkers,nthreads
    """
    # generate montblanc status string for use in log reports below
    if montblanc_threads is None:
        montblanc = ""
    elif montblanc_threads:
        montblanc = ", --montblanc-threads {}".format(montblanc_threads)
    else:
        montblanc = ", unlimited Montblanc threads"
    if force_serial:
        cubical.kernels.num_omp_threads = nthreads
        if nthreads:
            nthreads = max(nthreads, montblanc_threads or 1)
            print("forcing single-process mode, {} OMP and/or Montblanc threads".format(nthreads), file=log(0, "blue"))
        elif montblanc_threads:
            nthreads = montblanc_threads
            print("forcing single-process mode, single thread{}".format(montblanc), file=log(0, "blue"))
        return False, 0, nthreads
    if nworkers and nthreads:
        print("multi-process mode: --dist-nworker {} (+1), --dist-nthread {}{}".format(nworkers, nthreads, montblanc), file=log(0, "blue"))
        return True, nworkers, nthreads
    if ncpu:
        cores = ncpu - (montblanc_threads or 1)
        if not nworkers and not nthreads:
            if max_workers:
                cores = min(cores, max_workers)
            print("multi-process mode: {}+1 workers, single thread{}".format(cores, montblanc), file=log(0, "blue"))
            return True, cores, 1
        if nworkers:
            if max_workers:
                nworkers = min(nworkers, max_workers)
            nthreads = max(1, cores // nworkers)
            print("multi-process mode: --dist-nworker {} (+1), {} OMP threads{}".format(nworkers, nthreads, montblanc), file=log(0, "blue"))
            return True, nworkers, nthreads
        if nthreads:
            nworkers = max(1, cores // nthreads)
            if max_workers:
                nworkers = min(nworkers, max_workers)
            print("multi-process mode: {}+1 workers, --dist-nthread {}{}".format(nworkers, nthreads, montblanc), file=log(0, "blue"))
            return True, nworkers, nthreads
    else:  # ncpu not set, and nworkers/nthreads not both set
        if nworkers:
            if max_workers:
                nworkers = min(nworkers, max_workers)
            print("multi-process mode: --dist-nworker {} (+1), single thread{}".format(nworkers, montblanc), file=log(0, "blue"))
            return True, nworkers, 1
        if nthreads:
            print("single-process mode: --dist-thread {}{}".format(nthreads, montblanc), file=log(0, "blue"))
            return False, 0, nthreads
        print("single-process, single-thread mode{}".format(montblanc), file=log(0, "blue"))
        return False, 0, 0
    raise RuntimeError("can't be here -- this is a bug!")


def setup_parallelism(ncpu, nworker, nthread, force_serial, affinity, io_affinity, main_affinity, use_montblanc, montblanc_threads, max_workers):
    """
    Sets up parallelism, affinities and other properties of worker processes.
    
    Args:
        ncpu (int):
            max number of cores to use
        nworker (int):
            Number of workers to run (excluding the I/O worker). If 0, determine automatically.
        nthread (int):
            Number of threads to run per worker. If 0, determine automatically.
        force_serial (bool):
            If True, force serial mode, disabling worker parallelism (threads can still be used)
        affinity (int or str or None):
            If None or empty string, all CPU affinity setting is disabled.
            An "N:M" string specifies allocating starting with core N, stepping by M. An int N specifies
            starting with core N, stepping by 1.
        io_affinity (int):
            If set, enables affinity setting on the I/O worker, and allocates that number of cores to it
            (which includes montblanc threads), overriding the montblanc_threads setting.
        main_affinity (bool or str):
            If set, allocates a separate core to the main process and sets its affinity. If set to "io", pins
            the main process to the same cores as allocated to the I/O worker. If not set, the main process
            is not pinned to any core.
        use_montblanc (bool):
            True if montblanc is being used to predict the model
        montblanc_threads:
            Number of threads to allocate to montblanc. If 0, it will be unlimited.

    Returns:
        True if parallel mode is invoked (i.e. workers are to be launched)
    """
    global num_workers
    parallel, num_workers, nthread = _setup_workers_and_threads(force_serial, ncpu, nworker, nthread,
                                                                montblanc_threads if use_montblanc else None, max_workers)

    # in serial mode, simply set the Montblanc and/or worker thread count, and return
    if not parallel:
        if nthread:
            cubical.kernels.num_omp_threads = nthread
            numba.set_num_threads(int(nthread))
            os.environ["OMP_NUM_THREADS"] = os.environ["OMP_THREAD_LIMIT"] = str(nthread)
        return False

    # TODO: check actual number of cores on the system, and throw an error if affinity settings exceed this
    # (at the moment we just get an error down the line from libgomp and/or taskset)

    # child processes will inherit this
    cubical.kernels.num_omp_threads = nthread

    numba.set_num_threads(int(nthread))

    # parse affinity argument
    if affinity != "" and affinity is not None:
        if type(affinity) is int or re.match("^[\d+]$", affinity):
            core = int(affinity)
            corestep = 1
        elif re.match("^(\d+):(\d+)$", affinity):
            core, corestep = list(map(int, affinity.split(":")))
        else:
            raise ValueError("invalid affinity setting '{}'".format(affinity))
    else:
        core = corestep = affinity = None

    # first, the I/O process
    props = worker_process_properties["Process-1"] = worker_process_properties["ForkProcess-1"] = \
        dict(label="io", environ={})

    # allocate cores to I/O process, if asked to pin it
    
    # for now, since we can't figure out this tensorflow affinity shit
    if affinity is not None and io_affinity and use_montblanc:
        io_affinity = None
        print("Montblanc currently does not support CPU affinity settings: ignoring --dist-pin-io", file=log(0,"red"))
        
    if affinity is not None and io_affinity:
        num_io_cores = montblanc_threads if use_montblanc else 1
        io_cores = list(range(core,core+num_io_cores*corestep,corestep))
        core = core + num_io_cores * corestep
        # if Montblanc is in use, affinity controlled by GOMP setting, else by taskset
        if use_montblanc:
            props["environ"]["GOMP_CPU_AFFINITY"] = " ".join(map(str,io_cores))
            props["environ"]["OMP_NUM_THREADS"] = props["environ"]["OMP_THREAD_LIMIT"] = str(montblanc_threads)
        else:
            props["taskset"] = ",".join(io_cores)
    # else just restrict Montblanc threads, if asked to
    else:
        io_cores = []
        if use_montblanc and montblanc_threads:
            props["environ"]["OMP_NUM_THREADS"] = props["environ"]["OMP_THREAD_LIMIT"] = str(montblanc_threads)

    # are we asked to pin the main process?
    if affinity is not None and main_affinity:
        if main_affinity == "io":
            if io_affinity:
                worker_process_properties["MainProcess"]["taskset"] = str(io_cores[0])
        else:
            worker_process_properties["MainProcess"]["taskset"] = str(core)
            core = core + corestep

    # create entries for subprocesses, and allocate cores
    for icpu in range(1, num_workers + 1):
        name = "Process-{}".format(icpu + 1)
        name2 = "ForkProcess-{}".format(icpu + 1)
        props = worker_process_properties[name] = worker_process_properties[name2] = \
            dict(label="x%02d" % icpu, num_omp_threads=nthread, environ={})
        if affinity is not None:
            props["taskset"] = str(core)
            # if OMP is in use, set affinities via gomp
            if nthread:
                worker_cores = list(range(core, core + nthread * corestep, corestep))
                core += nthread * corestep
                props["environ"]["GOMP_CPU_AFFINITY"] = " ".join(map(str, worker_cores))
            else:
                core += corestep
    return True


def run_process_loop(ms, _tile_list, load_model, single_chunk, solver_type, solver_opts, debug_opts, out_opts):

    """
    Runs the main loop. If debugging is set, or single_chunk mode is on, forces serial mode.
    Otherwise selects serial or parallel depending on previous call to setup_parallelism().

    Args:
        ms:
        nworker:
        nthread:
        load_model:
        single_chunk:
        debugging:
        solver_type:
        solver_opts:
        debug_opts:
        out_opts:
    Returns:
        Stats dictionary
    """

    # if worker processes are launched, this global is inherited and will be accessed
    # by the I/O worker
    global tile_list
    tile_list = _tile_list
    
    if num_workers:
        return _run_multi_process_loop(ms, load_model, solver_type, solver_opts, debug_opts, out_opts)
    else:
        return _run_single_process_loop(ms, load_model, single_chunk, solver_type, solver_opts, debug_opts, out_opts)


def _run_multi_process_loop(ms, load_model, solver_type, solver_opts, debug_opts, out_opts):
    """
    Runs the main loop in multiprocessing mode.
    
    Args:
        ms: 
        nworker: 
        nthread: 
        load_model: 
        single_chunk: 
        solver_type: 
        solver_opts: 
        debug_opts: 
        out_opts:
    Returns:
        Stats dictionary
    """
    global tile_list

    # this accumulates SolverStats objects from each chunk, for summarizing later
    stats_dict = {}

    def reap_children():
        pid, status, _ = os.wait3(os.WNOHANG)
        if pid:
            print("child process {} exited with status {}. This is a bug, or an out-of-memory condition.".format(pid, status), file=log(0,"red"))
            print("This error is not recoverable: the main process will now commit ritual harakiri.", file=log(0,"red"))
            os._exit(1)
            raise RuntimeError("child process {} exited with status {}".format(pid, status))

    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor, \
            cf.ProcessPoolExecutor(max_workers=1) as io_executor:
        ms.flush()
        # this will be a dict of tile number: future loading that tile
        io_futures = {}
        # schedule I/O job to load tile 0
        io_futures[0] = io_executor.submit(_io_handler, load=0, save=None, load_model=load_model, out_opts=out_opts)
        # all I/O will be done by the I/O thread, so we need to close the MS in the main process
        # and reopen it afterwards
        ms.close()

        # now that the I/O child is forked, init main process properties (affinity etc.)
        # (Montblanc is finicky about affinities apparently, so we don't do it before)
        _init_worker(main=True)

        for itile, tile in enumerate(tile_list):
            # wait for I/O job on current tile to finish
            print("waiting for I/O on {}".format(tile.label), file=log(0))
            # have a timeout so that if a child process dies, we at least find out
            done = False
            while not done:
                reap_children()
                done, not_done = cf.wait([io_futures[itile]], timeout=10)

            # check if result was successful
            if not io_futures[itile].result():
                raise RuntimeError("I/O job on {} failed".format(tile.label))
            del io_futures[itile]

            # immediately schedule I/O job to save previous/load next tile
            load_next = itile + 1 if itile < len(tile_list) - 1 else None
            save_prev = itile - 1 if itile else None
            if load_next is not None or save_prev is not None:
                io_futures[itile + 1] = io_executor.submit(_io_handler, load=load_next,
                                                           save=save_prev, load_model=load_model)

            # submit solver jobs
            solver_futures = {}

            print("submitting solver jobs for {}".format(tile.label), file=log(0))

            for key in tile.get_chunk_keys():
                solver_futures[executor.submit(solver.run_solver, solver_type, itile, key, solver_opts, debug_opts)] = key
                print("submitted solver job for chunk {}".format(key), file=log(3))

            # wait for solvers to finish
            while solver_futures:
                reap_children()
                done, not_done = cf.wait(list(solver_futures.keys()), timeout=1)
                for future in done:
                    key = solver_futures[future]
                    stats = future.result()
                    stats_dict[tile.get_chunk_indices(key)] = stats
                    print("handled result of chunk {}".format(key), file=log(3))
                    del solver_futures[future]

            print("finished processing {}".format(tile.label), file=log(0))

        # ok, at this stage we've iterated over all the tiles, but there's an outstanding
        # I/O job saving the second-to-last tile (which was submitted with itile+1), and the last tile was
        # never saved, so submit a job for that (also to close the MS), and wait
        io_futures[-1] = io_executor.submit(_io_handler, load=None, save=-1, finalize=True)
        done = False
        while not done:
            reap_children()
            done, not_done = cf.wait([io_futures[-1]], timeout=10)

        # get flagcounts from result of job
        ms.update_flag_counts(io_futures[-1].result()['flagcounts'])

        # and reopen the MS again
        ms.reopen()

        return stats_dict


def _run_single_process_loop(ms, load_model, single_chunk, solver_type, solver_opts, debug_opts, out_opts):
    """
    Runs the main loop in single-CPU mode.

    Args:
        ms: 
        nworker: 
        nthread: 
        load_model: 
        single_chunk: 
        solver_type: 
        solver_opts: 
        debug_opts: 
        out_opts: output options from GD
    Returns:
        Stats dictionary
    """
    global tile_list

    stats_dict = {}

    for itile, tile in enumerate(tile_list):
        tile.load(load_model=load_model)
        processed = False
        for key in tile.get_chunk_keys():
            if not single_chunk or key == single_chunk:
                processed = True
                stats = solver.run_solver(solver_type, itile, key, solver_opts, debug_opts)
                stats_dict[tile.get_chunk_indices(key)] = stats

        if processed:
            only_save = ["output", "model", "weight", "flag", "bitflag"] if out_opts is None or out_opts["apply-solver-flags"] else \
                        ["output", "model", "weight"]
            tile.save(final=tile is tile_list[-1], only_save=only_save)
            for sd in tile.iterate_solution_chunks():
                solver.gm_factory.save_solutions(sd)
                solver.ifrgain_machine.accumulate(sd)
        else:
            print("  single-chunk {} not in this tile, skipping it.".format(single_chunk), file=log(0))
        tile.release(final=(itile == len(tile_list) - 1))
        # break out after single chunk is processed
        if processed and single_chunk:
            print("single-chunk {} was processed in this tile. Will now finish".format(single_chunk), file=log(0, "red"))
            break
    solver.ifrgain_machine.save()
    solver.gm_factory.close()

    return stats_dict

# this will be set to True after _init_worker is called()
_worker_initialized = None

def _init_worker(main=False):
    """
    Called inside every worker process first thing, to initialize its properties
    """
    global _worker_initialized
    # for the main process, do not set the singleton guard, as child processes will inherit it
    if not _worker_initialized:
        if not main:
            _worker_initialized = True

        name = multiprocessing.current_process().name
        if name not in worker_process_properties:
            print("WARNING: unrecognized worker process name '{}'. " \
                                "Please inform the developers.".format(name), file=log(0, "red"))
            return
        props = worker_process_properties[name]

        label = props.get("label")
        if label is not None:
            logger.set_subprocess_label(label)

        taskset = props.get("taskset")
        if taskset is not None:
            print("pid {}, setting CPU affinity to {} with taskset".format(os.getpid(), taskset), file=log(1,"blue"))
            os.system("taskset -pc {} {} >/dev/null".format(taskset, os.getpid()))

        environ = props.get("environ")
        if environ:
            os.environ.update(environ)
            for key, value in environ.items():
                print("setting {}={}".format(key, value), file=log(1,"blue"))

        num_omp_threads = props.get("num_omp_threads")
        if num_omp_threads is not None:
            print("enabling {} OMP threads".format(num_omp_threads), file=log(1,"blue"))
            import cubical.kernels
            cubical.kernels.num_omp_threads = num_omp_threads


def _io_handler(save=None, load=None, load_model=True, finalize=False, out_opts=None):
    """
    Handles disk reads and writes for the multiprocessing case.

    Args:
        save (None or int, optional):
            If specified, corresponds to index of Tile to save.
        load (None or int, optional):
            If specified, corresponds to index of Tile to load.
        load_model (bool, optional):
            If specified, loads model column from measurement set.
        finalize (bool, optional):
            If True, this is the last tile. Call the unlock method on the handler, and finalize everything else.
        out_opts (dict):
            Output options from GD
    Returns:
        bool:
            True if load/save was successful.
    """
    global tile_list
    try:
        _init_worker()
        result = {'success': True}
        if save is not None:
            tile = tile_list[save]
            itile = list(range(len(tile_list)))[save]
            print("saving {}".format(tile.label), file=log(0, "blue"))
            only_save = ["output", "model", "weight", "flag", "bitflag"] if out_opts is None or out_opts["apply-solver-flags"] else \
                        ["output", "model", "weight"]
            tile.save(final=finalize, only_save=only_save)
            for sd in tile.iterate_solution_chunks():
                solver.gm_factory.save_solutions(sd)
                solver.ifrgain_machine.accumulate(sd)
            if finalize:
                solver.ifrgain_machine.save()
                solver.gm_factory.close()
                result['flagcounts'] = tile.dh.flagcounts
            tile.release(final=(itile == len(tile_list) - 1))
        if load is not None:
            tile = tile_list[load]
            print("loading {}".format(tile.label), file=log(0, "blue"))
            tile.load(load_model=load_model)
        print("I/O job(s) complete", file=log(0, "blue"))
        return result
    except Exception as exc:
        print("I/O handler for load {} save {} failed with exception: {}".format(load, save, exc), file=log(0, "red"))
        print(traceback.format_exc(), file=log)
        raise

