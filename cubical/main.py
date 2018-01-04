# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Main code body. Handles options, invokes solvers and manages multiprocessing.
"""

## OMS: this workaround should not be necessary, now that https://github.com/ratt-ru/CubiCal/issues/75 is fixed
# import logging
# if 'vext' in logging.Logger.manager.loggerDict.keys():
#     for handler in logging.root.handlers:
#         logging.root.removeHandler(handler)
#     logging.getLogger('vext').setLevel(logging.WARNING)
##

import cPickle
import os
import os.path
import traceback
import sys
import re
import numpy as np
from time import time

# This is to keep matplotlib from falling over when no DISPLAY is set (which it otherwise does,
# even if one is only trying to save figures to .png.
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import concurrent.futures as cf

from cubical.tools import logger
# set the base name of the logger. This must happen before any other loggers are instantiated
# (Thus before anything else that uses the logger is imported!)
logger.init("cc")

import cubical
import cubical.data_handler as data_handler
from cubical.data_handler import DataHandler, Tile
from cubical.tools import parsets, dynoptparse, shm_utils, ModColor
from cubical.machines import machine_types
from cubical.machines import jones_chain_machine
from cubical.machines import ifr_gain_machine


log = logger.getLogger("main")

import cubical.solver as solver
import cubical.plots as plots
import cubical.flagging as flagging

from cubical.statistics import SolverStats

GD = None

class UserInputError(Exception):
    pass

# set to true with --Debug-Pdb 1, causes pdb to be invoked on exception
enable_pdb = True

def debug():
    """ Calls the main() function in debugging mode. """

    main(debugging=True)

def main(debugging=False):
    """
    Main cubical driver function. Reads options, sets up MS and solvers, calls the solver, etc.

    Args:
        debugging (bool, optional):
            If True, run in debugging mode.

    Raises:
        UserInputError:
            If neither --model-lsm nor --model-column were specified.
        UserInputError:
            If no Jones terms are enabled.
        UserInputError:
            If --out-mode is invalid.
        ValueError:
            If unknown Jones type is specified.
        RuntimeError:
            If I/O job on a tile failed.
    """

    # this will be set below if a custom parset is specified on the command line
    parset_file = None
    # "GD" is a global defaults dict, containing options set up from parset + command line
    global GD, enable_pdb

    try:
        if debugging:
            print>> log, "initializing from cubical.last"
            GD = cPickle.load(open("cubical.last"))
        else:
            default_parset = parsets.Parset("%s/DefaultParset.cfg" % os.path.dirname(__file__))

            def parse_command_line():
                parser = dynoptparse.DynamicOptionParser(usage='Usage: %prog [parset file] <options>',
                    description="""Questions, bug reports, suggestions: https://github.com/ratt-ru/CubiCal""",
                    version='%prog version {}'.format(cubical.VERSION),
                    defaults=default_parset.value_dict,
                    attributes=default_parset.attr_dict)
                parser.read_input()
                return parser

            parser = parse_command_line()

            positional_args = parser.get_arguments()
            # if a single argument is given, treat it as a parset and see if we can read it
            if len(positional_args) == 1:
                parset_file = positional_args[0]
                parset = parsets.Parset(parset_file)
                if not parset.success:
                    raise UserInputError("{} must be a valid parset file. Use -h for help.".format(parset_file))
                # update default parameters with values from parset
                default_parset.update_values(parset, newval=False)
                # re-read command-line options, since defaults will have been updated by the parset
                parser = parse_command_line()
            elif len(positional_args):
                raise UserInputError("{} must be a valid parset file. Use -h for help.".format(parset_file))

            # "GD" is a global defaults dict, containing options set up from parset + command line
            GD = parser.get_config()
            cPickle.dump(GD, open("cubical.last", "w"))

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
            if parset_file and os.path.exists(parset_file) and os.path.exists(save_parset) and \
                    os.path.samefile(save_parset, parset_file):
                basename = "~" + basename
                save_parset = basename + ".parset"
                print>> log, ModColor.Str(
                    "Your --Output-Name would overwrite its own parset. Using %s instead." % basename)
            parser.write_to_parset(save_parset)

        enable_pdb = GD["debug"]["pdb"]
        # clean up shared memory from any previous runs
        shm_utils.cleanupStaleShm()

        # now setup logging
        logger.logToFile(basename + ".log", append=GD["log"]["append"])
        logger.enableMemoryLogging(GD["log"]["memory"])
        logger.setBoring(GD["log"]["boring"])
        logger.setGlobalVerbosity(GD["log"]["verbose"])
        logger.setGlobalLogVerbosity(GD["log"]["file-verbose"])

        if not debugging:
            print>>log, "started " + " ".join(sys.argv)
        # print current options
        parser.print_config(dest=log)

        double_precision = GD["sol"]["precision"] == 64

        solver_type = GD['out']['mode']
        if solver_type not in solver.SOLVERS:
            raise UserInputError("invalid setting --out-mode {}".format(solver_type))
        solver_mode_name = solver.SOLVERS[solver_type].__name__.replace("_", " ")
        print>>log,ModColor.Str("mode: {}".format(solver_mode_name), col='green')
        # these flags are used below to tweak the behaviour of gain machines and model loaders
        apply_only = solver.SOLVERS[solver_type] in (solver.correct_only, solver.correct_residuals)
        load_model = solver.SOLVERS[solver_type] is not solver.correct_only   # no model needed in "correct only" mode

        if load_model and not GD["model"]["list"]:
            raise UserInputError("--model-list must be specified")

        ms = DataHandler(GD["data"]["ms"],
                              GD["data"]["column"], 
                              GD["model"]["list"].split(","),
                              output_column=GD["out"]["column"],
                              reinit_output_column=GD["out"]["reinit-column"],
                              taql=GD["sel"]["taql"],
                              fid=GD["sel"]["field"], 
                              ddid=GD["sel"]["ddid"],
                              channels=GD["sel"]["chan"],
                              flagopts=GD["flags"],
                              double_precision=double_precision,
                              weights=GD["weight"]["column"].split(","),
                              beam_pattern=GD["model"]["beam-pattern"], 
                              beam_l_axis=GD["model"]["beam-l-axis"], 
                              beam_m_axis=GD["model"]["beam-m-axis"],
                              active_subset=GD["sol"]["subset"],
                              min_baseline=GD["sol"]["min-bl"],
                              max_baseline=GD["sol"]["max-bl"],
                              use_ddes=GD["model"]["ddes"],
                              mb_opts=GD["montblanc"])

        data_handler.global_handler = ms

        # set up RIME

        solver_opts = GD["sol"]
        sol_jones = solver_opts["jones"]
        if type(sol_jones) is str:
            sol_jones = set(sol_jones.split(','))
        jones_opts = [GD[j.lower()] for j in sol_jones]
        # collect list of options from enabled Jones matrices
        if not len(jones_opts):
            raise UserInputError("No Jones terms are enabled")
        print>> log, ModColor.Str("Enabling {}-Jones".format(",".join(sol_jones)), col="green")

        # With a single Jones term, create a gain machine factory based on its type.
        # With multiple Jones, create a ChainMachine factory

        if len(jones_opts) == 1:
            jones_opts = jones_opts[0]
            # create a gain machine factory
            jones_class = machine_types.get_machine_class(jones_opts['type'])
            if jones_class is None:
                raise UserInputError("unknown Jones type '{}'".format(jones_opts['type']))
        else:
            jones_class = jones_chain_machine.JonesChain

        # set up subtraction options
        solver_opts["subtract-model"] = smod = GD["out"]["subtract-model"]
        if smod < 0 or smod >= len(ms.models):
            raise UserInputError("--out-subtract-model {} out of range for {} model(s)".format(smod, len(ms.models)))
        
        # parse subtraction directions as a slice or list
        subdirs = GD["out"]["subtract-dirs"]
        if type(subdirs) is int:
            subdirs = [subdirs]
        if subdirs:
            if type(subdirs) is str:
                try:
                    if ',' in subdirs:
                        subdirs = map(int, subdirs.split(","))
                    else:
                        subdirs = eval("np.s_[{}]".format(subdirs))
                except:
                    raise UserInputError("invalid --out-subtract-model option '{}'".format(subdirs))
            elif type(subdirs) is not list:
                raise UserInputError("invalid --out-subtract-dirs option '{}'".format(subdirs))
            # check ranges
            if type(subdirs) is list:
                out_of_range = [ d for d in subdirs if d < 0 or d >= len(ms.model_directions) ]
                if out_of_range:
                    raise UserInputError("--out-subtract-dirs {} out of range for {} model direction(s)".format(
                            ",".join(map(str, out_of_range)), len(ms.model_directions)))
            print>>log(0),"subtraction directions set to {}".format(subdirs)
        else:
            subdirs = slice(None)
        solver_opts["subtract-dirs"] = subdirs

        # create gain machine factory
        # TODO: pass in proper antenna and correlation names, rather than number
        grid = dict(ant=ms.antnames, corr=ms.feeds, time=ms.uniq_times, freq=ms.all_freqs)
        solver.gm_factory = jones_class.create_factory(grid=grid,
                                                       apply_only=apply_only,
                                                       double_precision=double_precision,
                                                       global_options=GD, jones_options=jones_opts)
                                                       
        # create IFR-based gain machine. Only compute gains if we're loading a model
        # (i.e. not in load-apply mode)
        solver.ifrgain_machine = ifr_gain_machine.IfrGainMachine(solver.gm_factory, GD["bbc"], compute=load_model)
        
        # set up chunking
        chunk_by = GD["data"]["chunk-by"]
        if type(chunk_by) is str:
            chunk_by = chunk_by.split(",")
        jump = float(GD["data"]["chunk-by-jump"])

        print>>log, "defining chunks (time {}, freq {}{})".format(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
            ", also when {} jumps > {}".format(", ".join(chunk_by), jump) if chunk_by else "")
        ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
                        chunk_by=chunk_by, chunk_by_jump=jump,
                        min_chunks_per_tile=max(GD["dist"]["ncpu"], GD["dist"]["min-chunks"]))


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
                tile.load(load_model=load_model)
                processed = False
                for key in tile.get_chunk_keys():
                    if not single_chunk or key == single_chunk:
                        processed = True
                        stats_dict[tile.get_chunk_indices(key)] = \
                            solver.run_solver(solver_type, itile, key, solver_opts)
                if processed:
                    tile.save()
                    for sd in tile.iterate_solution_chunks():
                        solver.gm_factory.save_solutions(sd)
                        solver.ifrgain_machine.accumulate(sd)
                    print>>log(0),ModColor.Str("single-chunk {} was processed in this tile. Will now finish".format(single_chunk))
                else:
                    print>>log(0),"  single-chunk {} not in this tile, skipping it.".format(single_chunk)
                tile.release()
                # break out after single chunk is processed
                if processed and single_chunk:
                    break
            solver.ifrgain_machine.save()
            solver.gm_factory.close()

        else:
            with cf.ProcessPoolExecutor(max_workers=ncpu-1) as executor, \
                 cf.ProcessPoolExecutor(max_workers=1) as io_executor:

                ms.flush()
                # this will be a dict of tile number: future loading that tile
                io_futures = {}
                # schedule I/O job to load tile 0
                io_futures[0] = io_executor.submit(_io_handler, load=0, save=None)
                # all I/O will be done by the io_executor, so we need to close the MS in the main process
                # and reopen it afterwards
                ms.close()
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
                    io_futures[itile+1] = io_executor.submit(_io_handler, load=load_next,
                                                             save=save_prev, load_model=load_model)

                    # submit solver jobs
                    solver_futures = {}

                    print>>log(0),"submitting solver jobs for tile {}".format(itile)

                    for key in tile.get_chunk_keys():
                        if not single_chunk or key == single_chunk:
                            solver_futures[executor.submit(solver.run_solver, solver_type,
                                                           itile, key, solver_opts)] = key
                            print>> log(3), "submitted solver job for chunk {}".format(key)

                    # wait for solvers to finish
                    for future in cf.as_completed(solver_futures):
                        key = solver_futures[future]
                        stats = future.result()
                        stats_dict[tile.get_chunk_indices(key)] = stats
                        print>>log(3),"handled result of chunk {}".format(key)

                    print>> log(0), "done with tile {}".format(itile)

                # ok, at this stage we've iterated over all the tiles, but there's an outstanding
                # I/O job saving the second-to-last tile (which was submitted with itile+1), and the last tile was
                # never saved, so submit a job for that (also to close the MS), and wait
                io_futures[-1] = io_executor.submit(_io_handler, load=None, save=-1, finalize=True)
                cf.wait(io_futures.values())

                # and reopen the MS again
                ms.reopen()

        print>>log, ModColor.Str("Time taken for {}: {} seconds".format(solver_mode_name, time() - t0), col="green")

        if not apply_only:
            # now summarize the stats
            print>> log, "computing summary statistics"
            st = SolverStats(stats_dict)
            filename = basename + ".stats.pickle"
            st.save(filename)
            print>> log, "saved summary statistics to %s" % filename

            # flag based on summary stats
            flag3 = flagging.flag_chisq(st, GD, basename, ms.nddid_actual)

            if flag3 is not None:
                st.apply_flagcube(flag3)
                if GD["flags"]["save"] and flag3.any() and not GD["data"]["single-chunk"]:
                    print>>log,"regenerating output flags based on post-solution flagging"
                    flagcol = ms.flag3_to_col(flag3)
                    ms.save_flags(flagcol)

            # make plots
            if GD["out"]["plots"]:
                plots.make_summary_plots(st, ms, GD, basename)

        ms.close()

        print>>log, ModColor.Str("completed successfully", col="green")

    except Exception, exc:
        import traceback
        print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__, 
                                                                    exc, traceback.format_exc()))
        if enable_pdb and not type(exc) is UserInputError:
            from cubical.tools import pdb
            exc, value, tb = sys.exc_info()
            pdb.post_mortem(tb)
        sys.exit(2 if type(exc) is UserInputError else 1)

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
    
    try:
        if save is not None:
            tile = Tile.tile_list[save]
            itile = range(len(Tile.tile_list))[save]
            print>>log(0, "blue"),"saving tile {}/{}".format(itile, len(Tile.tile_list))
            tile.save(unlock=finalize)
            for sd in tile.iterate_solution_chunks():
                solver.gm_factory.save_solutions(sd)
                solver.ifrgain_machine.accumulate(sd)
            if finalize:
                solver.ifrgain_machine.save()
                solver.gm_factory.close()
            tile.release()
        if load is not None:
            print>>log(0, "blue"),"loading tile {}/{}".format(load, len(Tile.tile_list))
            Tile.tile_list[load].load(load_model=load_model)
        return True
    except Exception, exc:
        print>> log, ModColor.Str("I/O handler for load {} save {} failed with exception: {}".format(load, save, exc))
        print>> log, traceback.format_exc()
        raise

