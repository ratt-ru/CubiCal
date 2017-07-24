import cPickle
import os
import os.path
import traceback
import sys
from time import time

import concurrent.futures as cf

import cubical.data_handler as data_handler
from cubical.data_handler import ReadModelHandler, Tile
from cubical.tools import logger, parsets, myoptparse, shm_utils, ModColor

log = logger.getLogger("main")

import cubical.solver as solver
import cubical.plots as plots
import cubical.flagging as flagging


from cubical.statistics import SolverStats



def init_options(parset, savefile=None):
    """
    Creates an command-line option parser, populates it based on the content of the given Parset object,
    and parses the command line.

    If savefile is set, dumps the option settings to savefile.

    Returns the option parser.
    """

    default_values = parset.value_dict
    attrs = parset.attr_dict

    desc = """Questions and suggestions: RATT"""

    OP = myoptparse.MyOptParse(usage='Usage: %prog [parset file] <options>', version='%prog version 0.1',
                               description=desc, defaults=default_values, attributes=attrs)

    # create options based on contents of parset
    for section in parset.sections:
        values = default_values[section]
        # "_Help" value in each section is its documentation string
        OP.OptionGroup(values.get("_Help", section), section)
        for name, value in default_values[section].iteritems():
            if not attrs[section][name].get("no_cmdline"):
                OP.add_option(name, value)

    OP.Finalise()
    OP.ReadInput()

    if savefile:
        cPickle.dump(OP, open(savefile,"w"))

    return OP




# set to true with --Debug-Pdb 1, causes pdb to be invoked on exception
enable_pdb = False

def debug():
    """
    This calls the main() function in debugging mode.
    """
    main(debugging=True)

def main(debugging=False):
    """
    Main cubical driver function.

    Reads options, sets up MS and solvers, calls the solver, etc.
    """
    # cl;ean up shared memory from any previous runs
    shm_utils.cleanupStaleShm()

    # init logger
    logger.enableMemoryLogging(True)

    # this will be set below if a custom parset is specified on the command line
    parset_file = None
    # "GD" is a global defaults dict, containing options set up from parset + command line
    global GD, enable_pdb

    try:
        if debugging:
            print>> log, "initializing from cubical.last"
            optparser = cPickle.load(open("cubical.last"))
            # "GD" is a global defaults dict, containing options set up from parset + command line
            GD = optparser.DicoConfig
        else:
            default_parset = parsets.Parset("%s/DefaultParset.cfg" % os.path.dirname(__file__))
            optparser = init_options(default_parset, "cubical.last")

            positional_args = optparser.GiveArguments()
            # if a single argument is given, treat it as a parset and see if we can read it
            if len(positional_args) == 1:
                parset_file = positional_args[0]
                parset = parsets.Parset(parset_file)
                if not parset.success:
                    optparser.ExitWithError("%s must be a valid parset file. Use -h for help."%parset_file)
                    sys.exit(1)
                # update default parameters with values from parset
                default_parset.update_values(parset, newval=False)
                # re-read command-line options, since defaults will have been updated by the parset
                optparser = init_options(default_parset, "cubical.last")
            elif len(positional_args):
                optparser.ExitWithError("Incorrect number of arguments. Use -h for help.")
                sys.exit(1)

            # "GD" is a global defaults dict, containing options set up from parset + command line
            GD = optparser.DicoConfig

            # get basename for all output files
            basename = GD["out"]["name"]
            if not basename:
                basename = "out"

            # create directory for output files, if it doesn't exist
            dirname = os.path.dirname(basename)
            if not os.path.exists(dirname) and not dirname == "":
                os.mkdir(dirname)

            # save parset with all settings. We refuse to clobber a parset with itself
            # (so e.g. "gocubical test.parset --Section-Option foo" does not overwrite test.parset)
            save_parset = basename + ".parset"
            if parset_file and os.path.exists(parset_file) and os.path.samefile(save_parset, parset_file):
                basename = "~" + basename
                save_parset = basename + ".parset"
                print>> log, ModColor.Str(
                    "Your --Output-Name would overwrite its own parset. Using %s instead." % basename)
            optparser.ToParset(save_parset)

        enable_pdb = GD["debug"]["pdb"]

        # now setup logging
        logger.logToFile(basename + ".log", append=GD["log"]["append"])
        logger.enableMemoryLogging(GD["log"]["memory"])
        if not debugging:
            print>>log, "started " + " ".join(sys.argv)
        # print current options
        optparser.Print(dest=log)

        # enable verbosity
        logger.verbosity = GD["debug"]["verbose"]

        ms = ReadModelHandler(GD["data"]["ms"], 
                              GD["data"]["column"], 
                              GD["model"]["lsm"], 
                              GD["model"]["column"],
                              output_column=GD["out"]["column"],
                              taql=GD["sel"]["taql"],
                              fid=GD["sel"]["field"], 
                              ddid=GD["sel"]["ddid"],
                              flagopts=GD["flags"],
                              precision=GD["sol"]["precision"],
                              ddes=GD["model"]["ddes"],
                              weight_column=GD["weight"]["column"],
                              beam_pattern=GD["model"]["beam-pattern"], 
                              beam_l_axis=GD["model"]["beam-l-axis"], 
                              beam_m_axis=GD["model"]["beam-m-axis"])

        data_handler.global_handler = ms

        print>>log, "defining chunks"
        ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
                        min_chunks_per_tile=max(GD["dist"]["ncpu"], GD["dist"]["min-chunks"]))

        saving_data = True
        if GD["out"]["vis"] == "corrected":
            solver_type = 'solve-correct'
        elif GD["out"]["vis"] == "residuals":
            solver_type = 'solve-residual'
        else:
            solver_type = 'solve'

        solver_opts = GD["sol"]

        t0 = time()

        # Debugging mode: run serially if processes is not set, or if a single chunk is specified.
        # Normal mode: use futures to run in parallel. TODO: Figure out if we can used shared memory to
        # improve performance.
        ncpu = GD["dist"]["ncpu"]

        # this accumulates SolverStats objects from each chunk, for summarizing later
        stats_dict = {}

        single_chunk = GD["data"]["single-chunk"]

        # target function has the following signature/behaviour
        # inputs: itile:       number of tile (in ms.tile_list)
        #         key:         chunk key (as returned by tile.get_chunk_keys())
        #         solver_opts: dict of solver options
        # returns: stats object

        if debugging or ncpu <= 1 or single_chunk:
            for itile, tile in enumerate(Tile.tile_list):
                tile.load()
                for key in tile.get_chunk_keys():
                    if not single_chunk or tile.get_chunk_label(key) == single_chunk:
                        stats_dict[key] = solver.run_solver(solver_type, itile, key, solver_opts)
                tile.save()
                    # ms.add_to_gain_dict(outdict['gains'], chunk_info,
                    #                     GD["sol"]["time-int"], GD["sol"]["freq-int"])
                tile.release()

        else:
            # all I/O will be done by the io_executor, so we need to release the locks
            ms.unlock()

            with cf.ProcessPoolExecutor(max_workers=ncpu-1) as executor, \
                 cf.ProcessPoolExecutor(max_workers=1) as io_executor:

                # this will be a dict of tile number: future loading that tile
                io_futures = {}
                # schedule I/O job to load tile 0
                io_futures[0] = io_executor.submit(_io_handler, load=0, save=None)
                for itile, tile in enumerate(Tile.tile_list):
                    # wait for I/O job on current tile to finish
                    print>>log(0),"waiting for I/O on tile {}".format(itile)
                    done, not_done = cf.wait([io_futures[itile]])
                    if not done or not io_futures[itile].result():
                        raise RuntimeError("I/O job on tile {} failed".format(itile))
                    del io_futures[itile]

                    # immediately schedule I/O job to save previous/load next tile
                    print>>log(0),"scheduling I/O on tile {}".format(itile+1)

                    load_next = itile+1 if itile < len(Tile.tile_list)-1 else None
                    save_prev = itile-1 if itile else None
                    io_futures[itile+1] = io_executor.submit(_io_handler, load=load_next, save=save_prev)

                    # submit solver jobs
                    solver_futures = {}

                    print>>log(0),"submitting solver jobs for tile {}".format(itile)

                    for key in tile.get_chunk_keys():
                        if not single_chunk or tile.get_chunk_label(key) == single_chunk:
                            solver_futures[executor.submit(solver.run_solver, solver_type, itile, key, solver_opts)] = key
                            print>> log(3), "submitted solver job for chunk {}".format(tile.get_chunk_label(key))

                    # wait for solvers to finish
                    for future in cf.as_completed(solver_futures):
                        key = solver_futures[future]
                        stats = future.result()
                        stats_dict[key] = stats
                        print>>log(3),"handled result of chunk {}".format(tile.get_chunk_label(key))

                    print>> log(0), "done with tile {}".format(itile)

                # ok, at this stage we've iterated over all the tiles, but there's an outstanding
                # I/O job saving the second-to-last tile (which was submitted with itile+1), and the last tile was
                # never saved, so submit a job for that (also to close the MS), and wait
                io_futures[-1] = io_executor.submit(_io_handler, load=None, save=-1, unlock=True)
                cf.wait(io_futures.values())

        print>>log, ModColor.Str("Time taken for solve: {} seconds".format(time() - t0), col="green")
        ms.lock()

        # now summarize the stats
        print>> log, "computing summary statistics"
        st = SolverStats(stats_dict)
        filename = basename + ".stats.pickle"
        st.save(filename)
        print>> log, "saved summary statistics to %s" % filename

        # flag based on summary stats
        flag3 = flagging.flag_chisq(st, GD, basename, ms.nddid)

        if flag3 is not None:
            st.apply_flagcube(flag3)
            if GD["flags"]["save"] and flag3.any() and not GD["data"]["single-chunk"]:
                print>>log,"regenerating output flags based on post-solution flagging"
                flagcol = ms.flag3_to_col(flag3)
                ms.save_flags(flagcol)

        # make plots
        if GD["out"]["plots"]:
            plots.make_summary_plots(st, GD, basename)

        ms.write_gain_dict()
        ms.close()

        print>>log, ModColor.Str("completed successfully", col="green")

    except Exception, exc:
        import traceback
        print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc()))
        if enable_pdb:
            import pdb
            exc, value, tb = sys.exc_info()
            pdb.post_mortem(tb)  # more "modern"
        sys.exit(1)

def _io_handler(save=None, load=None, unlock=False):
    try:
        if save is not None:
            Tile.tile_list[save].save(unlock)
            Tile.tile_list[save].release()
        if load is not None:
            Tile.tile_list[load].load()
        return True
    except Exception, exc:
        print>> log, ModColor.Str("I/O handler for load {} save {} failed with exception: {}".format(load, save, exc))
        print>> log, traceback.format_exc()
        raise

