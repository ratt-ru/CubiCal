from ReadModelHandler import *
from time import time
import os, os.path
import sys
import cPickle
import concurrent.futures as cf
from Tools import logger, parsets, myoptparse
log = logger.getLogger("main")


import solver
from statistics import SolverStats


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
    Main cubecal driver function.

    Reads options, sets up MS and solvers, calls the solver, etc.
    """
    # init logger
    logger.enableMemoryLogging(True)

    # this will be set below if a custom parset is specified on the command line
    parset_file = None
    # "GD" is a global defaults dict, containing options set up from parset + command line
    global GD, enable_pdb

    try:
        if debugging:
            print>> log, "initializing from cubecal.last"
            optparser = cPickle.load(open("cubecal.last"))
            # "GD" is a global defaults dict, containing options set up from parset + command line
            GD = optparser.DicoConfig
        else:
            default_parset = parsets.Parset("%s/DefaultParset.cfg" % os.path.dirname(__file__))
            optparser = init_options(default_parset, "cubecal.last")

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
                optparser = init_options(default_parset, "cubecal.last")
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
            # (so e.g. "gocubecal test.parset --Section-Option foo" does not overwrite test.parset)
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

        ddid, ddid_to = GD["sel"]["ddid"], GD["sel"]["ddid-to"]
        if ddid is not None and ddid_to is not None:
            ddid = ddid, ddid_to+1

        ms = ReadModelHandler(GD["data"]["ms"], GD["data"]["column"], GD["model"]["lsm"], GD["model"]["column"],
                              taql=GD["sel"]["taql"],
                              fid=GD["sel"]["field"], ddid=ddid,
                              precision=GD["sol"]["precision"],
                              ddes=GD["model"]["ddes"],
                              weight_column=GD["weight"]["column"])
        ms.apply_flags = bool(GD["flags"]["apply"])
        ms.bitmask = GD["flags"]["apply-bitmask"]

        print>>log, "reading MS columns"
        ms.mass_fetch()
        print>>log, "defining chunks"
        ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["freq-chunk"], single_chunk_id=GD["data"]["single-chunk"])

        if GD["out"]["vis"] == "corrected":
            target = solver.solve_and_correct
        elif GD["out"]["vis"] == "residuals":
            target = solver.solve_and_correct_res
        else:
            target = solver.solve_gains

        solver_opts = GD["sol"]

        t0 = time()

        # Debugging mode: run serially if processes is not set, or if a single chunk is specified.
        # Normal mode: use futures to run in parallel. TODO: Figure out if we can used shared memory to
        # improve performance.
        ncpu = GD["dist"]["ncpu"]

        # this accumulates SolverStats objects from each chunk, for summarizing later
        stats_dict = {}

        if debugging or ncpu <= 1 or GD["data"]["single-chunk"]:
            for obser, model, flags, weight, tfkey, chunk_label in ms:
                gm, covis, stats = target(obser, model, flags, solver_opts, label = chunk_label)
                stats_dict[tfkey] = stats
                if covis is not None:
                    ms.arr_to_col(covis, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f])

                ms.add_to_gain_dict(gm.gains, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f],
                                    GD["sol"]["time-int"], GD["sol"]["freq-int"])

        else:
            with cf.ProcessPoolExecutor(max_workers=ncpu) as executor:
                future_gains = { executor.submit(target, obser, model, flags, solver_opts, label=chunk_label) :
                                 [tfkey, ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f]
                                 for obser, model, flags, weight, tfkey, chunk_label in ms }

                for future in cf.as_completed(future_gains):
                    gm, covis, stats = future.result()
                    stats_dict[future_gains[future][0]] = stats
                    if covis is not None:
                        ms.arr_to_col(covis, future_gains[future][1:])

                    ms.add_to_gain_dict(gm.gains, future_gains[future][1:],
                                        GD["sol"]["time-int"], GD["sol"]["freq-int"])

        print>>log, ModColor.Str("Time taken: {} seconds".format(time() - t0), col="green")

        # now summarize the stats
        if GD["out"]["plots"]:
            # summarize stats
            print>>log,"computing summary statistics"
            st = SolverStats(stats_dict)

            import pylab

            # these control the layout of saved plots
            DPI = 150.   # resolution: determines size of text relative to plots
            ZOOM = 10    # image zoom. Roughly determines size of "pixels" when rendering images
            def save_figure(name, width, height):
                pylab.gcf().set_size_inches(min(width, 10000/DPI), min(height, 10000/DPI))
                filename = "{}.{}.png".format(basename, name)
                pylab.savefig(filename, dpi=DPI)
                print>>log, "saved "+filename
                if GD["out"]["plots-show"]:
                    pylab.show()

            # plot noise per time/channel
            pylab.subplot(121)
            pylab.title("Noise on input data")
            pylab.xlabel("channel")
            pylab.ylabel("timeslot")
            pylab.imshow(np.sqrt(st.timechan.dv2))
            pylab.colorbar()
            pylab.subplot(122)
            pylab.title("Noise on residuals")
            pylab.xlabel("channel")
            pylab.ylabel("timeslot")
            pylab.imshow(np.sqrt(st.timechan.dr2))
            pylab.colorbar()
            nt,nf = st.timechan.dv2.shape
            save_figure("noise.tf", nf*ZOOM/DPI*2.5, nt*ZOOM/DPI*1.1)

            # plot noise per antenna/channel
            pylab.subplot(121)
            noise = np.sqrt(st.chanant.dv2)
            noise[noise==0] = np.inf
            nf,nant = noise.shape
            for ant in xrange(nant):
                pylab.plot(noise[:,ant],'o-')
            pylab.title("Noise by antenna".format(ddid))
            pylab.xlabel("channel")
            pylab.ylabel("noise")
            pylab.subplot(122)
            for chan in xrange(nf):
                pylab.plot(noise[chan,:],'o-')
            pylab.title("Noise by channel".format(ddid))
            pylab.xlabel("antenna")
            pylab.ylabel("noise")
            save_figure("noise.antchan", 10, 5)

        ms.write_gain_dict()

        if target is not solver.solve_gains:
            print>>log, "saving visibilities to {}".format(GD["out"]["column"])
            ms.save(ms.covis, GD["out"]["column"])
    except Exception, exc:
        import traceback
        print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc()))
        if enable_pdb:
            import pdb
            exc, value, tb = sys.exc_info()
            pdb.post_mortem(tb)  # more "modern"
        sys.exit(1)
