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
from __future__ import print_function
from builtins import range
from six import string_types
from future.moves import pickle
import os, os.path
import sys
import warnings
import numpy as np
import re
import datetime
import getpass
import traceback
from time import time
from cubical import VERSION

# This is to keep matplotlib from falling over when no DISPLAY is set (which it otherwise does,
# even if one is only trying to save figures to .png.
import matplotlib

from cubical.tools import logger
# set the base name of the logger. This must happen before any other loggers are instantiated
# (Thus before anything else that uses the logger is imported!)
logger.init("cc")

# Some modules cause issues with logging - grab their loggers and 
# manually set the log levels to something less annoying.

import logging
import warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

GD = None

# the getuser() call fails under singularity --contain, so fall back to $USER
try:
    _user = getpass.getuser()
except:
    _user = os.environ.get("USER", "unknown")

_start_datetime = datetime.datetime.now()

_runtime_templates = dict(DATE=_start_datetime.strftime("%Y%m%d"),
                          TIME=_start_datetime.strftime("%H%M%S"),
                          USER=_user,
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
            for i in range(10):
                name1 = name.format(**keys)
                if name1 == name:
                    break
                name = name1
        return name
    except Exception as exc:
        print("{}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc()), file=log)
        if name == name0:
            print(ModColor.Str("Error substituting '{}', see above".format(name)), file=log)
        else:
            print(ModColor.Str("Error substituting '{}' (derived from '{}'), see above".format(name, name0)), file=log)
        raise ValueError(name)

from cubical.data_handler.ms_data_handler import MSDataHandler
from cubical.data_handler.wisdom import estimate_mem
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

    # keep a list of messages here, until we have a logfile open
    prelog_messages = []

    def prelog_print(level, message):
        prelog_messages.append((level, message))

    print("Using CubiCal version {}.".format(VERSION), file=log)

    try:
        if debugging:
            print("initializing from cubical.last", file=log)
            GD = pickle.load(open("cubical.last"))
            basename = GD["out"]["name"]
            parser = None
        else:
            default_parset = parsets.Parset("%s/DefaultParset.cfg" % os.path.dirname(__file__))

            # if first argument is a filename, treat it as a parset

            if len(sys.argv) > 1 and not sys.argv[1][0].startswith('-'):
                custom_parset_file = sys.argv[1]
                print("reading defaults from {}".format(custom_parset_file), file=log)
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
            outdir = expand_templated_name(GD["out"]["dir"]).strip()
            basename = expand_templated_name(GD["out"]["name"]).strip()
            can_overwrite = GD["out"]["overwrite"]
            can_backup = GD["out"]["backup"]

            explicit_basename_path = "/" in basename
            folder_is_ccout  = False

            if explicit_basename_path:
                prelog_print(0, "output basename explicitly set to {}, --out-dir setting ignored".format(basename))
                outdir = os.path.dirname(basename)
            elif outdir == "." or not outdir:
                outdir = None
                prelog_print(0, "using output basename {} in current directory".format(basename))
            else:
                # append implicit .cc-out suffix, unless already there (or ends with .cc-out)
                if not outdir.endswith("/"):
                    if outdir.endswith(".cc-out"):
                        outdir += "/"
                    else:
                        outdir += ".cc-out/"
                folder_is_ccout = outdir.endswith(".cc-out/")
                basename = outdir + basename
                if outdir != "/":
                    outdir = outdir.rstrip("/")
                prelog_print(0, "using output basename {}".format(basename))

            # create directory for output files, if specified, and it doesn't exist
            if outdir and not os.path.exists(outdir):
                prelog_print(0, "creating new output directory {}".format(outdir))
                os.mkdir(outdir)

            # are we going to be overwriting a previous run?
            out_parset = "{}.parset".format(basename)
            if os.path.exists(out_parset):
                prelog_print(0, "{} already exists, possibly from a previous run".format(out_parset))

                if can_backup:
                    if folder_is_ccout:
                        # find non-existing directory name for backup
                        backup_dir = outdir + ".0"
                        N = 0
                        while os.path.exists(backup_dir):
                            N += 1
                            backup_dir = "{}.{}".format(outdir, N)
                        # rename old directory, if we ended up manipulating the directory name
                        os.rename(outdir, backup_dir)
                        os.mkdir(outdir)
                        prelog_print(0, ModColor.Str("backed up existing {} to {}".format(outdir, backup_dir), "blue"))
                    else:
                        prelog_print(0, "refusing to auto-backup output directory, since it is not a .cc-out dir")

                if os.path.exists(out_parset):
                    if can_overwrite:
                        prelog_print(0, "proceeding anyway since --out-overwrite is set")
                    else:
                        if folder_is_ccout:
                            prelog_print(0, "won't proceed without --out-overwrite and/or --out-backup")
                        else:
                            prelog_print(0, "won't proceed without --out-overwrite")
                        raise UserInputError("{} already exists: won't overwrite previous run".format(out_parset))

            GD["out"]["name"] = basename

            # "GD" is a global defaults dict, containing options set up from parset + command line
            pickle.dump(GD, open("cubical.last", "wb"))

            # save parset with all settings
            parser.write_to_parset(out_parset)

        enable_pdb = GD["debug"]["pdb"]

        if GD["debug"]["escalate-warnings"]:
            warnings.simplefilter('error', UserWarning)
            np.seterr(all='raise')
            if GD["debug"]["escalate-warnings"] > 1:
                warnings.simplefilter('error', Warning)
                log(0).print("all warnings will be escalated to exceptions")
            else:
                log(0).print("UserWarnings will be escalated to exceptions")

        # clean up shared memory from any previous runs
        shm_utils.cleanupStaleShm()

        # now setup logging
        logger.logToFile(basename + ".log", append=GD["log"]["append"])
        logger.enableMemoryLogging(GD["log"]["memory"])
        logger.setBoring(GD["log"]["boring"])
        logger.setGlobalVerbosity(GD["log"]["verbose"])
        logger.setGlobalLogVerbosity(GD["log"]["file-verbose"])

        if not debugging:
            print("started " + " ".join(sys.argv), file=log)

        # dump accumulated messages from before log was open
        for level, message in prelog_messages:
            print(message, file=log(level))
        prelog_messages = []

        # clean up shared memory from any previous runs
        shm_utils.cleanupStaleShm()

        # disable matplotlib's tk backend if we're not going to be showing plots
        if GD['out']['plots'] =='show' or GD['madmax']['plot'] == 'show':
            import pylab
            try:
                pylab.figure()
                pylab.close()
            except Exception as exc:
                import traceback
                print(ModColor.Str("Error initializing matplotlib: {}({})\n {}".format(type(exc).__name__,
                                                                                       exc, traceback.format_exc())), file=log)
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
        out_opts = GD["out"]
        sol_jones = solver_opts["jones"]
        if isinstance(sol_jones, string_types):
            sol_jones = set(sol_jones.split(','))
        jones_opts = [GD[j.lower()] for j in sol_jones]
        # collect list of options from enabled Jones matrices
        if not len(jones_opts):
            raise UserInputError("No Jones terms are enabled")
        print(ModColor.Str("Enabling {}-Jones".format(",".join(sol_jones)), col="green"), file=log)

        have_dd_jones = any([jo['dd-term'] for jo in jones_opts])
        have_solvables = any([jo['solvable'] for jo in jones_opts])

        solver.GD = GD

        # set up data handler

        solver_type = GD['out']['mode']
        if solver_type not in solver.SOLVERS:
            raise UserInputError("invalid setting --out-mode {}".format(solver_type))
        solver_mode_name = solver.SOLVERS[solver_type].__name__.replace("_", " ")
        print(ModColor.Str("mode: {}".format(solver_mode_name), col='green'), file=log)
        # these flags are used below to tweak the behaviour of gain machines and model loaders
        apply_only = solver.SOLVERS[solver_type].is_apply_only
        print("solver is apply-only type: {}".format(apply_only), file=log(0))
        load_model = solver.SOLVERS[solver_type].is_model_required
        print("solver requires model: {}".format(load_model), file=log(0))
        
        if not apply_only and not have_solvables:
            raise UserInputError("No Jones terms have been marked as solvable")

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
                           active_subset=GD["sol"]["subset"] if not apply_only else None,
                           min_baseline=GD["sol"]["min-bl"] if not apply_only else 0,
                           max_baseline=GD["sol"]["max-bl"] if not apply_only else 0,
                           chunk_freq=GD["data"]["freq-chunk"],
                           rebin_freq=GD["data"]["rebin-freq"],
                           do_load_CASA_kwtables = GD["out"]["casa-gaintables"],
                           feed_rotate_model=GD["model"]["feed-rotate"],
                           pa_rotate_model=GD["model"]["pa-rotate"],
                           pa_rotate_montblanc=GD["montblanc"]["pa-rotate"],
                           derotate_output=GD["out"]["derotate"],
                           do_normalize_data=GD["data"]["normalize"]
                           )

        solver.metadata = ms.metadata
        # if using dual-corr mode, propagate this into Jones options
        if ms.ncorr == 2:
            for jo in jones_opts:
                jo['diag-only'] = True
                jo['diag-data'] = True
            solver_opts['diag-only'] = True
            solver_opts['diag-data'] = True

        # With a single Jones term, create a gain machine factory based on its type.
        # With multiple Jones, create a ChainMachine factory
        term_iters = solver_opts["term-iters"]
        if type(term_iters) is int:
            term_iters = [term_iters] * len(jones_opts)
            solver_opts["term-iters"] = term_iters
            len(jones_opts) > 1 and log.warn("Multiple gain terms specified, but a recipe of solver sol-term-iters not given. "
                                             "This may indicate user error. We will assume doing the same number of iterations per term and "
                                             "stopping on the last term on the chain.")
        elif type(term_iters) is list and len(term_iters) == 1:
            term_iters = term_iters * len(jones_opts)
            solver_opts["term-iters"] = term_iters
            len(jones_opts) > 1 and log.warn("Multiple gain terms specified, but a recipe of solver sol-term-iters not given. "
                                             "This may indicate user error. We will assume doing the same number of iterations per term and "
                                             "stopping on the last term on the chain.")
        elif type(term_iters) is list and len(term_iters) < len(jones_opts):
            raise ValueError("sol-term-iters is a list, but does not match or exceed the number of gain terms being solved. "
                             "Please either only set a single value to be used or provide a list to construct a iteration recipe")
        elif type(term_iters) is list and len(term_iters) >= len(jones_opts):
            pass # user is executing a recipe
        else:
            raise TypeError("sol-term-iters is neither a list, nor a int. Check your parset")

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
        elif jones_opts[0]['type'].startswith("robust"):
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
                       use_ddes=have_dd_jones and dde_mode != 'never',
                       degrid_opts=GD["degridding"])

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
                if isinstance(subdirs, string_types):
                    try:
                        if ',' in subdirs:
                            subdirs = list(map(int, subdirs.split(",")))
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
                print("subtraction directions set to {}".format(subdirs), file=log(0))
            else:
                subdirs = slice(None)
            solver_opts["subtract-dirs"] = subdirs

        # create gain machine factory
        # TODO: pass in proper antenna and correlation names, rather than number
        solver_opts["correct-dir"] = GD["out"]["correct-dir"] if GD["out"]["correct-dir"] >= 0 else None

        grid = dict(dir=list(range(len(ms.model_directions) or 1)), ant=ms.antnames, corr=list(ms.feeds), time=ms.uniq_times, freq=ms.all_freqs)
        solver.gm_factory = jones_class.create_factory(grid=grid,
                                                       apply_only=apply_only,
                                                       double_precision=double_precision,
                                                       global_options=GD, jones_options=jones_opts)

        solver.gm_factory.set_metadata(ms)
                                                       
        # create IFR-based gain machine. Only compute gains if we're loading a model
        # (i.e. not in load-apply mode)
        solver.ifrgain_machine = ifr_gain_machine.IfrGainMachine(solver.gm_factory, GD["bbc"], compute=load_model)

        solver.legacy_version12_weights = GD["weight"]["legacy-v1-2"]

        single_chunk = GD["data"]["single-chunk"]
        single_tile = GD["data"]["single-tile"]


        # set up chunking

        chunk_by = GD["data"]["chunk-by"]
        if isinstance(chunk_by, string_types):
            chunk_by = chunk_by.split(",")
        jump = float(GD["data"]["chunk-by-jump"])

        if single_chunk:
            chunks_per_tile = 1
            max_chunks_per_tile = 1
        else:
            nw = GD["dist"]["nworker"] or ((GD["dist"]["ncpu"] / (GD["dist"]["nthread"] or 1)) or 1) - 1
            chunks_per_tile = max(GD["dist"]["min-chunks"], nw, 1)
            max_chunks_per_tile = 0
            if GD["dist"]["max-chunks"]:
                chunks_per_tile = max(max(GD["dist"]["max-chunks"], chunks_per_tile), 1)
                max_chunks_per_tile = GD["dist"]["max-chunks"]

        print("defining chunks (time {}, freq {}{})".format(GD["data"]["time-chunk"], GD["data"]["freq-chunk"],
            ", also when {} jumps > {}".format(", ".join(chunk_by), jump) if chunk_by else ""), file=log)

        chunks_per_tile, tile_list = ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["rebin-time"],
                                            GD["data"]["freq-chunk"],
                                            chunk_by=chunk_by, chunk_by_jump=jump,
                                            chunks_per_tile=chunks_per_tile, max_chunks_per_tile=max_chunks_per_tile)

        # setup worker process properties
        workers.setup_parallelism(GD["dist"]["ncpu"], GD["dist"]["nworker"], GD["dist"]["nthread"],
                                  debugging or single_chunk,
                                  GD["dist"]["pin"], GD["dist"]["pin-io"], GD["dist"]["pin-main"],
                                  ms.use_montblanc, GD["montblanc"]["threads"],
                                  max_workers=chunks_per_tile)


        # Estimate memory usage. This is still experimental.
        estimate_mem(ms, tile_list, GD["data"], GD["dist"])

        # now that we have tiles, define the flagging situation (since this may involve a one-off iteration through the
        # MS to populate the column)
        ms.define_flags(tile_list, flagopts=GD["flags"])

        # single-chunk implies single-tile
        if single_tile >= 0:
            tile_list = tile_list[single_tile:single_tile+1]
            print("--data-single-tile {} set, will process only the one tile".format(single_tile), file=log(0, "blue"))
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
                print("--data-single-chunk {} in {}, rows {}:{}".format(
                    single_chunk, tile.label, min(rc.rows0), max(rc.rows0)+1), file=log(0, "blue"))
            else:
                raise ValueError("--data-single-chunk {}: chunk with this ID not found".format(single_chunk))

        # run the main loop

        t0 = time()

        stats_dict = workers.run_process_loop(ms, tile_list, load_model, single_chunk, solver_type, solver_opts, debug_opts, out_opts)


        print(ModColor.Str("Time taken for {}: {} seconds".format(solver_mode_name, time() - t0), col="green"), file=log)

        # print flagging stats
        print(ModColor.Str("Flagging stats: ",col="green") + " ".join(ms.get_flag_counts()), file=log)

        if not apply_only:
            # now summarize the stats
            print("computing summary statistics", file=log)
            st = SolverStats(stats_dict)
            filename = basename + ".stats.pickle"
            st.save(filename)
            print("saved summary statistics to %s" % filename, file=log)
            print_stats = GD["log"]["stats"]
            if print_stats:
                print("printing some summary statistics below", file=log(0))
                thresholds = []
                for thr in GD["log"]["stats-warn"].split(","):
                    field, value = thr.split(":")
                    thresholds.append((field, float(value)))
                    print("  highlighting {}>{}".format(field, float(value)), file=log(0))
                if print_stats == "all":
                    print_stats = st.get_notrivial_chunk_statfields()
                else:
                    print_stats = print_stats.split("//")
                for stats in print_stats:
                    if stats[0] != "{":
                        stats = "{{{}}}".format(stats)
                    lines = st.format_chunk_stats(stats, threshold=thresholds)
                    print("  summary stats for {}:\n  {}".format(stats, "\n  ".join(lines)), file=log(0))

            if GD["postmortem"]["enable"]:
                # flag based on summary stats
                flag3 = flagging.flag_chisq(st, GD, basename, ms.nddid_actual)

                if flag3 is not None:
                    st.apply_flagcube(flag3)
                    if GD["flags"]["save"] and flag3.any() and not GD["data"]["single-chunk"]:
                        print("regenerating output flags based on post-solution flagging", file=log)
                        flagcol = ms.flag3_to_col(flag3)
                        ms.save_flags(flagcol)

            # make plots
            if GD["out"]["plots"]:
                import cubical.plots
                try:
                    cubical.plots.make_summary_plots(st, ms, GD, basename)
                except Exception as exc:
                    if GD["debug"]["escalate-warnings"]:
                        raise
                    import traceback
                    print(file=ModColor.Str("An error has occurred while making summary plots: {}({})\n {}".format(type(exc).__name__,
                                                                                           exc,
                                                                                           traceback.format_exc())))
                    print(ModColor.Str("This is not fatal, but should be reported (and your plots have gone missing!)"), file=log)

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
                except Exception as exc:
                    import traceback
                    print(file=ModColor.Str("An error has occurred while making BBC plots: {}({})\n {}".format(type(exc).__name__,
                                                                                           exc,
                                                                                           traceback.format_exc())))
                    print(ModColor.Str("This is not fatal, but should be reported (and your plots have gone missing!)"), file=log)

        ms.close()

        print(ModColor.Str("completed successfully", col="green"), file=log)

    except RuntimeWarning:
        from cubical.tools import pdb
        pdb.set_trace()

    except Exception as exc:
        for level, message in prelog_messages:
            print(message, file=log(level))

        if type(exc) is UserInputError:
            print(ModColor.Str(exc), file=log)
        else:
            import traceback
            print(ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__,
                                                                    exc, traceback.format_exc())), file=log)
            if enable_pdb and type(exc) is not UserInputError:
                warnings.filterwarnings("default")  # in case pdb itself throws a warning
                from cubical.tools import pdb
                exc, value, tb = sys.exc_info()
                pdb.post_mortem(tb)
        sys.exit(2 if type(exc) is UserInputError else 1)
