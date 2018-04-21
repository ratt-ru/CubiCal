from cubical.data_handler import Tile



_worker_initialized = None

def _init_worker():
    """Called inside every worker process to initialize its properties"""
    global _worker_initialized
    if not _worker_initialized:
        _worker_initialized = True
        name = multiprocessing.current_process().name
        if name not in _worker_process_properties:
            print>> log(0, "red"), "WARNING: unrecognized worker process name '{}'. " \
                                "Please inform the developers.".format(name)
            return
        label, affinity, nthread, gomp = _worker_process_properties[name]
        logger.set_subprocess_label(label)
        if affinity is not None:
            print>>log(1),"setting worker process ({}) CPU affinity to {}".format(os.getpid(), affinity)
            os.system("taskset -pc {} {} >/dev/null".format(affinity, os.getpid()))
        if gomp is not None:
            os.environ["GOMP_CPU_AFFINITY"] = gomp
            print>>log(0,"red"),"set GOMP_CPU_AFFINITY={}".format(os.environ["GOMP_CPU_AFFINITY"])
        import cubical.kernels
        cubical.kernels.num_omp_threads = nthread


def _io_handler(save=None, load=None, load_model=True, finalize=False):
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
            If True, save will call the unlock method on the handler.

    Returns:
        bool:
            True if load/save was successful.
    """
    _init_worker()
    try:
        result = {'success': True}
        if save is not None:
            tile = Tile.tile_list[save]
            itile = range(len(Tile.tile_list))[save]
            print>>log(0, "blue"),"saving {}".format(tile.label)
            tile.save(unlock=finalize)
            for sd in tile.iterate_solution_chunks():
                solver.gm_factory.save_solutions(sd)
                solver.ifrgain_machine.accumulate(sd)
            if finalize:
                solver.ifrgain_machine.save()
                solver.gm_factory.close()
                result['flagcounts'] = tile.handler.flagcounts
            tile.release()
        if load is not None:
            tile = Tile.tile_list[load]
            print>>log(0, "blue"),"loading {}".format(tile.label)
            tile.load(load_model=load_model)
        print>> log(0, "blue"), "I/O job(s) complete"
        return result
    except Exception, exc:
        print>> log, ModColor.Str("I/O handler for load {} save {} failed with exception: {}".format(load, save, exc))
        print>> log, traceback.format_exc()
        raise

