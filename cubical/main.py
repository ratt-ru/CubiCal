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
import os, os.path
import sys
import warnings
import numpy as np
from time import time

# This is to keep matplotlib from falling over when no DISPLAY is set (which it otherwise does,
# even if one is only trying to save figures to .png.
import matplotlib

from cubical.tools import logger
# set the base name of the logger. This must happen before any other loggers are instantiated
# (Thus before anything else that uses the logger is imported!)
logger.init("cc")

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import cubical.data_handler as data_handler
from cubical.data_handler import DataHandler, Tile
from cubical.tools import parsets, dynoptparse, shm_utils, ModColor
from cubical.machines import machine_types
from cubical.machines import jones_chain_machine
from cubical.machines import ifr_gain_machine
from cubical import workers

log = logger.getLogger("main")

import cubical.solver as solver
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
    custom_parset_file = None
    # "GD" is a global defaults dict, containing options set up from parset + command line
    global GD, enable_pdb

    try:
        if debugging:
            print>> log, "initializing from cubical.last"
            GD = cPickle.load(open("cubical.last"))
            basename = GD["out"]["name"]
            parser = None
        else:
            default_parset = parsets.Parset("%s/DefaultParset.cfg" % os.path.dirname(__file__))

            # if first argument is a filename, treat it as a parset

            if len(sys.argv) > 1 and not sys.argv[1][0].startswith('-'):
                custom_parset_file = sys.argv[1]
                print>> log, "reading defaults from {}".format(custom_parset_file)
                try:
                    parset = parsets.Parset(custom_parset_file)
                except:
                    import traceback
                    traceback.print_exc()
                    raise UserInputError("'{}' must be a valid parset file. Use -h for help.".format(custom_parset_file))
                if not parset.success:
                    raise UserInputError("'{}' must be a valid parset file. Use -h for help.".format(custom_parset_file))
                # update default parameters with values from parset
                default_parset.update_values(parset, other_filename=' in {}'.format(custom_parset_file))

            import cubical
            parser = dynoptparse.DynamicOptionParser(usage='Usage: %prog [parset file] <options>',
                    description="""Questions, bug reports, suggestions: https://github.com/ratt-ru/CubiCal""",
                    version='%prog version {}'.format(cubical.VERSION),
                    defaults=default_parset.value_dict,
                    attributes=default_parset.attr_dict)

            # now read the full input from command line
            # "GD" is a global defaults dict, containing options set up from parset + command line
            GD = parser.read_input()

            # if a single argument is given, it should have been the parset
            if len(parser.get_arguments()) != (1 if custom_parset_file else 0):
                raise UserInputError("Unexpected number of arguments. Use -h for help.")

            # "GD" is a global defaults dict, containing options set up from parset + command line
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
            if custom_parset_file and os.path.exists(custom_parset_file) and os.path.exists(save_parset) and \
                    os.path.samefile(save_parset, custom_parset_file):
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

        # disable matplotlib's tk backend if we're not going to be showing plots
        if GD['out']['plots-show']:
            import pylab
            try:
                pylab.figure()
            except Exception, exc:
                import traceback
                print>>log, ModColor.Str("Error initializing matplotlib: {}({})\n {}".format(type(exc).__name__,
                                                                                       exc, traceback.format_exc()))
                raise UserInputError("matplotlib can't connect to X11. Suggest disabling --out-plots-show.")
        else:
            matplotlib.use("Agg")

        # print current options
        if parser is not None:
            parser.print_config(dest=log)

        double_precision = GD["sol"]["precision"] == 64

        # set up RIME

        solver_opts = GD["sol"]
        debug_opts  = GD["debug"]
        sol_jones = solver_opts["jones"]
        if type(sol_jones) is str:
            sol_jones = set(sol_jones.split(','))
        jones_opts = [GD[j.lower()] for j in sol_jones]
        # collect list of options from enabled Jones matrices
        if not len(jones_opts):
            raise UserInputError("No Jones terms are enabled")
        print>> log, ModColor.Str("Enabling {}-Jones".format(",".join(sol_jones)), col="green")

        have_dd_jones = any([jo['dd-term'] for jo in jones_opts])

        # TODO: in this case data_handler can be told to only load diagonal elements. Save memory!
        # top-level diag-diag enforced across jones terms
        if solver_opts['diag-diag']:
            for jo in jones_opts:
                jo['diag-diag'] = True
        else:
            solver_opts['diag-diag'] = all([jo['diag-diag'] for jo in jones_opts])

        # set up data handler

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
                          output_column=GD["out"]["column"],
                          output_model_column=GD["out"]["model-column"],
                          reinit_output_column=GD["out"]["reinit-column"],
                          taql=GD["sel"]["taql"],
                          fid=GD["sel"]["field"],
                          ddid=GD["sel"]["ddid"],
                          channels=GD["sel"]["chan"],
                          flagopts=GD["flags"],
                          diag=solver_opts["diag-diag"],
                          double_precision=double_precision,
                          beam_pattern=GD["model"]["beam-pattern"],
                          beam_l_axis=GD["model"]["beam-l-axis"],
                          beam_m_axis=GD["model"]["beam-m-axis"],
                          active_subset=GD["sol"]["subset"],
                          min_baseline=GD["sol"]["min-bl"],
                          max_baseline=GD["sol"]["max-bl"],
                          do_load_CASA_kwtables = GD["out"]["casa-gaintables"])
        
        data_handler.global_handler = ms

        # With a single Jones term, create a gain machine factory based on its type.
        # With multiple Jones, create a ChainMachine factory

        if len(jones_opts) == 1:
            jones_opts = jones_opts[0]
            # for just one term, propagate --sol-term-iters, if set, into its max-iter setting
            term_iters = solver_opts["term-iters"]
            if term_iters:
                jones_opts["max-iter"] = term_iters[0] if hasattr(term_iters,'__getitem__') else term_iters
            # create a gain machine factory
            jones_class = machine_types.get_machine_class(jones_opts['type'])
            if jones_class is None:
                raise UserInputError("unknown Jones type '{}'".format(jones_opts['type']))
        else:
            jones_class = jones_chain_machine.JonesChain

        # init models
        dde_mode = GD["model"]["ddes"]

        if dde_mode == 'always' and not have_dd_jones:
            raise UserInputError("we have '--model-ddes always', but no direction dependent Jones terms enabled")

        ms.init_models(GD["model"]["list"].split(","),
                       GD["weight"]["column"].split(","),
                       mb_opts=GD["montblanc"],
                       use_ddes=have_dd_jones and dde_mode != 'never')

        if len(ms.model_directions) < 2 and have_dd_jones and dde_mode == 'auto':
            raise UserInputError("--model-list does not specify directions. "
                    "Have you forgotten a @dE tag perhaps? Rerun with '--model-ddes never' to proceed anyway.")

        if load_model:
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
        

        single_chunk = GD["data"]["single-chunk"]

        # setup worker process properties

        workers.setup_parallelism(GD["dist"]["ncpu"], GD["dist"]["nworker"], GD["dist"]["nthread"],
                                  debugging or single_chunk,
                                  GD["dist"]["pin"], GD["dist"]["pin-io"], GD["dist"]["pin-main"],
                                  ms.use_montblanc, GD["montblanc"]["threads"])

        # set up chunking

        chunk_by = GD["data"]["chunk-by"]
        if type(chunk_by) is str:
            chunk_by = chunk_by.split(",")
        jump = float(GD["data"]["chunk-by-jump"])

        chunks_per_tile = max(GD["dist"]["min-chunks"], workers.num_workers, 1)
        if GD["dist"]["max-chunks"]:
            chunks_per_tile = max(GD["dist"]["max-chunks"], chunks_per_tile)

        print>>log, "defining chunks (time {}, freq {}{})".format(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
            ", also when {} jumps > {}".format(", ".join(chunk_by), jump) if chunk_by else "")

        chunks_per_tile = ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
                                            chunk_by=chunk_by, chunk_by_jump=jump,
                                            chunks_per_tile=chunks_per_tile, max_chunks_per_tile=GD["dist"]["max-chunks"])

        # run the main loop

        t0 = time()
        stats_dict = workers.run_process_loop(ms, load_model, single_chunk, solver_type, solver_opts, debug_opts)

        print>>log, ModColor.Str("Time taken for {}: {} seconds".format(solver_mode_name, time() - t0), col="green")

        # print flagging stats
        print>>log, ModColor.Str("Flagging stats: ",col="green") + " ".join(ms.get_flag_counts())

        if not apply_only:
            # now summarize the stats
            print>> log, "computing summary statistics"
            st = SolverStats(stats_dict)
            filename = basename + ".stats.pickle"
            st.save(filename)
            print>> log, "saved summary statistics to %s" % filename

            if GD["flags"]["post-sol"]:
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
                import cubical.plots as plots
                plots.make_summary_plots(st, ms, GD, basename)

        # make BBC plots
        if solver.ifrgain_machine and solver.ifrgain_machine.is_computing() and GD["out"]["plots"]:
            import cubical.plots.ifrgains
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.ComplexWarning)
                cubical.plots.ifrgains.make_ifrgain_plots(solver.ifrgain_machine.reload(), ms, GD, basename)

        ms.close()

        print>>log, ModColor.Str("completed successfully", col="green")

    except Exception, exc:
        import traceback
        if type(exc) is UserInputError:
            print>> log, ModColor.Str(exc)
        else:
            print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__,
                                                                    exc, traceback.format_exc()))
            if enable_pdb and not type(exc) is UserInputError:
                from cubical.tools import pdb
                exc, value, tb = sys.exc_info()
                pdb.post_mortem(tb)
        sys.exit(2 if type(exc) is UserInputError else 1)

