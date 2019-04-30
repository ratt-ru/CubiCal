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
import re
import datetime
import getpass
import traceback
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

GD = None

_start_datetime = datetime.datetime.now()

_runtime_templates = dict(DATE=_start_datetime.strftime("%Y%m%d"),
                          TIME=_start_datetime.strftime("%H%M%S"),
                          USER=getpass.getuser(),
                          HOST=os.uname()[1],
                          ENV=os.environ)

def expand_templated_name(name, **keys):
    """
        Helper method: expands name from templated name. This uses the standard
        str.format() function, passing in GD (global dict of options), as well as any keys supplied,
        as well as the _runtime_templates dict above.
        This allows for name templates that reference both the parset, as well as runtime conditions:
        e.g. "{data[ms]}-ddid{sel[ddid]}-{DATE}-{TIME}".

        Args:
            name (str):
                the templated name
            keys (optional):
                any optional substitution keys.

        Returns:
            str:
                Expanded filename
    """
    name0 = name
    try:
        if name:
            keys.update(_runtime_templates)
            keys.update(GD)
            # substitute recursively, but up to a limit
            for i in xrange(10):
                name1 = name.format(**keys)
                if name1 == name:
                    break
                name = name1
        return name
    except Exception, exc:
        print>> log, "{}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc())
        if name == name0:
            print>> log, ModColor.Str("Error substituting '{}', see above".format(name))
        else:
            print>> log, ModColor.Str("Error substituting '{}' (derived from '{}'), see above".format(name, name0))
        raise ValueError(name)

from cubical.data_handler.ms_data_handler import MSDataHandler
from cubical.tools import parsets, dynoptparse, shm_utils, ModColor
from cubical.machines import machine_types
from cubical.machines import jones_chain_machine, jones_chain_robust_machine
from cubical.machines import ifr_gain_machine
from cubical import workers

log = logger.getLogger("main")

import cubical.solver as solver
import cubical.flagging as flagging

from cubical.statistics import SolverStats


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

            # get dirname and basename for all output files
            basename = expand_templated_name(GD["out"]["name"])

            if not basename:
                dirname, basename = "cubical-out", "cubical"
            elif basename.endswith("/"):
                dirname, basename = basename[:-1], "cubical"
            elif "/" in basename:
                dirname, basename = os.path.split(basename)
                dirname = dirname.rstrip("/")
            else:
                dirname, basename = ".", basename

            # create directory for output files, if specified, and it doesn't exist
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            # find unique output name, if needed
            if os.path.exists("{}/{}.log".format(dirname, basename)) and not GD["out"]["overwrite"]:
                print>> log(0, "blue"), "{}/{}.log already exists, won't overwrite".format(dirname, basename)
                dirname0, basename0 = dirname, basename
                N = -1
                while os.path.exists("{}/{}.log".format(dirname, basename)):
                    N += 1
                    if dirname == ".":
                        basename = "{}.{}".format(basename0, N)
                    else:
                        dirname = "{}.{}".format(dirname0, N)
                # rename old directory, if we ended up manipulating the directory name
                if dirname != dirname0:
                    os.rename(dirname0, dirname)
                    print>> log(0, "blue"), "saved previous {} to {}".format(dirname0, dirname)
                    dirname = dirname0
                    os.mkdir(dirname)

            if dirname != ".":
                basename = "{}/{}".format(dirname, basename)
            print>> log(0, "blue"), "using {} as base for output files".format(basename)

            GD["out"]["name"] = basename

            # "GD" is a global defaults dict, containing options set up from parset + command line
            cPickle.dump(GD, open("cubical.last", "w"))

            # save parset with all settings. We refuse to clobber a parset with itself
            # (so e.g. "gocubical test.parset --Section-Option foo" does not overwrite test.parset)
            save_parset = basename + ".parset"
            if custom_parset_file and os.path.exists(custom_parset_file) and os.path.exists(save_parset) and \
                    os.path.samefile(save_parset, custom_parset_file):
                basename = "~" + basename
                save_parset = basename + ".parset"
                print>> log, ModColor.Str("your --out-name would overwrite its own parset. Using {} instead.".format(basename))
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
        if GD['out']['plots'] =='show' or GD['madmax']['plot'] == 'show':
            import pylab
            try:
                pylab.figure()
                pylab.close()
            except Exception, exc:
                import traceback
                print>>log, ModColor.Str("Error initializing matplotlib: {}({})\n {}".format(type(exc).__name__,
                                                                                       exc, traceback.format_exc()))
                raise UserInputError("matplotlib can't connect to X11. Can't use --out-plots show or --madmax-plot show.")
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

        # set up data handler

        solver_type = GD['out']['mode']
        if solver_type not in solver.SOLVERS:
            raise UserInputError("invalid setting --out-mode {}".format(solver_type))
        solver_mode_name = solver.SOLVERS[solver_type].__name__.replace("_", " ")
        print>>log,ModColor.Str("mode: {}".format(solver_mode_name), col='green')
        # these flags are used below to tweak the behaviour of gain machines and model loaders
        apply_only = solver.SOLVERS[solver_type] in (solver.correct_only, solver.correct_residuals, solver.subtract_only)
        load_model = solver.SOLVERS[solver_type] is not solver.correct_only   # no model needed in "correct only" mode

        if load_model and not GD["model"]["list"]:
            raise UserInputError("--model-list must be specified")

        ms = MSDataHandler(GD["data"]["ms"],
                           GD["data"]["column"],
                           output_column=GD["out"]["column"],
                           output_model_column=GD["out"]["model-column"],
                           output_weight_column=GD["out"]["weight-column"],
                           reinit_output_column=GD["out"]["reinit-column"],
                           taql=GD["sel"]["taql"],
                           fid=GD["sel"]["field"],
                           ddid=GD["sel"]["ddid"],
                           channels=GD["sel"]["chan"],
                           diag=GD["sel"]["diag"],
                           beam_pattern=GD["model"]["beam-pattern"],
                           beam_l_axis=GD["model"]["beam-l-axis"],
                           beam_m_axis=GD["model"]["beam-m-axis"],
                           active_subset=GD["sol"]["subset"],
                           min_baseline=GD["sol"]["min-bl"],
                           max_baseline=GD["sol"]["max-bl"],
                           chunk_freq=GD["data"]["freq-chunk"],
                           rebin_freq=GD["data"]["rebin-freq"],
                           do_load_CASA_kwtables = GD["out"]["casa-gaintables"],
                           feed_rotate_model=GD["model"]["feed-rotate"],
                           pa_rotate_model=GD["model"]["pa-rotate"],
                           pa_rotate_montblanc=GD["montblanc"]["pa-rotate"],
                           derotate_output=GD["out"]["derotate"],
                           )

        # if using dual-corr mode, propagate this into Jones options
        if ms.ncorr == 2:
            for jo in jones_opts:
                jo['diag-only'] = True
                jo['diag-data'] = True
            solver_opts['diag-only'] = True
            solver_opts['diag-data'] = True

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
        elif jones_opts[0]['type'] == "robust-2x2":
            jones_class = jones_chain_robust_machine.JonesChain
        else:
            jones_class = jones_chain_machine.JonesChain

        # init models
        dde_mode = GD["model"]["ddes"]

        if dde_mode == 'always' and not have_dd_jones:
            raise UserInputError("we have '--model-ddes always', but no direction dependent Jones terms enabled")

        # force floats in Montblanc calculations
        mb_opts = GD["montblanc"]
        # mb`_opts['dtype'] = 'float'

        ms.init_models(str(GD["model"]["list"]).split(","),
                       GD["weight"]["column"].split(",") if GD["weight"]["column"] else None,
                       fill_offdiag_weights=GD["weight"]["fill-offdiag"],
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

        solver.GD = GD
        solver.metadata = ms.metadata

        grid = dict(ant=ms.antnames, corr=ms.feeds, time=ms.uniq_times, freq=ms.all_freqs)
        solver.gm_factory = jones_class.create_factory(grid=grid,
                                                       apply_only=apply_only,
                                                       double_precision=double_precision,
                                                       global_options=GD, jones_options=jones_opts)
                                                       
        # create IFR-based gain machine. Only compute gains if we're loading a model
        # (i.e. not in load-apply mode)
        solver.ifrgain_machine = ifr_gain_machine.IfrGainMachine(solver.gm_factory, GD["bbc"], compute=load_model)

        solver.legacy_version12_weights = GD["weight"]["legacy-v1-2"]

        single_chunk = GD["data"]["single-chunk"]
        single_tile = GD["data"]["single-tile"]

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

        chunks_per_tile, tile_list = ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["rebin-time"],
                                            GD["data"]["freq-chunk"],
                                            chunk_by=chunk_by, chunk_by_jump=jump,
                                            chunks_per_tile=chunks_per_tile, max_chunks_per_tile=GD["dist"]["max-chunks"])

        # now that we have tiles, define the flagging situation (since this may involve a one-off iteration through the
        # MS to populate the column)
        ms.define_flags(tile_list, flagopts=GD["flags"])

        # single-chunk implies single-tile
        if single_tile >= 0:
            tile_list = tile_list[single_tile:single_tile+1]
            print>> log(0, "blue"), "--data-single-tile {} set, will process only the one tile".format(single_tile)
        elif single_chunk:
            match = re.match("D([0-9]+)T([0-9]+)", single_chunk)
            if not match:
                raise ValueError("invalid setting: --data-single-chunk {}".format(single_chunk))
            ddid_tchunk = int(match.group(1)), int(match.group(2))

            tilemap = { (rc.ddid, rc.tchunk): (tile, rc) for tile in tile_list for rc in tile.rowchunks }
            single_tile_rc = tilemap.get(ddid_tchunk)
            if single_tile_rc:
                tile, rc = single_tile_rc
                tile_list = [tile]
                print>> log(0, "blue"), "--data-single-chunk {} in {}, rows {}:{}".format(
                    single_chunk, tile.label, min(rc.rows0), max(rc.rows0)+1)
            else:
                raise ValueError("--data-single-chunk {}: chunk with this ID not found".format(single_chunk))

        # run the main loop

        t0 = time()

        stats_dict = workers.run_process_loop(ms, tile_list, load_model, single_chunk, solver_type, solver_opts, debug_opts)


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
            print_stats = GD["log"]["stats"]
            if print_stats:
                print>> log(0), "printing some summary statistics below"
                thresholds = []
                for thr in GD["log"]["stats-warn"].split(","):
                    field, value = thr.split(":")
                    thresholds.append((field, float(value)))
                    print>>log(0), "  highlighting {}>{}".format(field, float(value))
                if print_stats == "all":
                    print_stats = st.get_notrivial_chunk_statfields()
                else:
                    print_stats = print_stats.split("//")
                for stats in print_stats:
                    if stats[0] != "{":
                        stats = "{{{}}}".format(stats)
                    lines = st.format_chunk_stats(stats, threshold=thresholds)
                    print>>log(0),"  summary stats for {}:\n  {}".format(stats, "\n  ".join(lines))

            if GD["postmortem"]["enable"]:
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
                import cubical.plots
                try:
                    cubical.plots.make_summary_plots(st, ms, GD, basename)
                except Exception, exc:
                    if GD["debug"]["escalate-warnings"]:
                        raise
                    import traceback
                    print>> ModColor.Str("An error has occurred while making summary plots: {}({})\n {}".format(type(exc).__name__,
                                                                                           exc,
                                                                                           traceback.format_exc()))
                    print>>log, ModColor.Str("This is not fatal, but should be reported (and your plots have gone missing!)")

        # make BBC plots
        if solver.ifrgain_machine and solver.ifrgain_machine.is_computing() and GD["bbc"]["plot"] and GD["out"]["plots"]:
            import cubical.plots.ifrgains
            if GD["debug"]["escalate-warnings"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", np.ComplexWarning)
                    cubical.plots.ifrgains.make_ifrgain_plots(solver.ifrgain_machine.reload(), ms, GD, basename)
            else:
                try:
                    cubical.plots.ifrgains.make_ifrgain_plots(solver.ifrgain_machine.reload(), ms, GD, basename)
                except Exception, exc:
                    import traceback
                    print>> ModColor.Str("An error has occurred while making BBC plots: {}({})\n {}".format(type(exc).__name__,
                                                                                           exc,
                                                                                           traceback.format_exc()))
                    print>>log, ModColor.Str("This is not fatal, but should be reported (and your plots have gone missing!)")

        ms.close()

        print>>log, ModColor.Str("completed successfully", col="green")

    except Exception, exc:
        if type(exc) is UserInputError:
            print>> log, ModColor.Str(exc)
        else:
            import traceback
            print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__,
                                                                    exc, traceback.format_exc()))
            if enable_pdb and not type(exc) is UserInputError:
                from cubical.tools import pdb
                exc, value, tb = sys.exc_info()
                pdb.post_mortem(tb)
        sys.exit(2 if type(exc) is UserInputError else 1)

