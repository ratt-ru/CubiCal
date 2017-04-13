from ReadModelHandler import *
from time import time
import math
import cyfull_complex as cyfull
import os, os.path
import sys
import cPickle
import concurrent.futures as cf
from Tools import logger, parsets, myoptparse
log = logger.getLogger("full_complex")

verbose = 0

def compute_js(obser_arr, model_arr, gains, t_int=1, f_int=1):
    """
    This function computes the (J^H)R term of the GN/LM method for the
    full-polarisation, phase-only case.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        gains (np.array): Array containing the current gain estimates.

    Returns:
        jhr (np.array): Array containing the result of computing (J^H)R.
    """

    n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

    jh = np.zeros_like(model_arr)

    cyfull.cycompute_jh(model_arr, gains, jh, t_int, f_int)

    jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

    jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

    # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
    if n_dir > 1:
        r = compute_residual(obser_arr, model_arr, gains, t_int, f_int)
    else:
        r = obser_arr

    cyfull.cycompute_jhr(jh, r, jhr, t_int, f_int)

    jhj = np.zeros(jhr_shape, dtype=obser_arr.dtype)

    cyfull.cycompute_jhj(jh, jhj, t_int, f_int)

    jhjinv = np.empty(jhr_shape, dtype=obser_arr.dtype)

    cyfull.cycompute_jhjinv(jhj, jhjinv)

    return jhr, jhjinv

def compute_update(model_arr, obser_arr, gains, t_int=1, f_int=1):
    """
    This function computes the update step of the GN/LM method. This is
    equivalent to the complete (((J^H)J)^-1)(J^H)R.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        gains (np.array): Array containing the current gain estimates.
        jhjinv (np.array): Array containing (J^H)J)^-1. (Invariant)

    Returns:
        update (np.array): Array containing the result of computing
            (((J^H)J)^-1)(J^H)R
    """


    jhr, jhjinv = compute_js(obser_arr, model_arr, gains, t_int, f_int)

    update = np.empty_like(jhr)

    cyfull.cycompute_update(jhr, jhjinv, update)

    return update


def compute_residual(obser_arr, model_arr, resid_arr, gains, t_int=1, f_int=1):
    """
    This function computes the residual. This is the difference between the
    observed data, and the model data with the gains applied to it.

    Args:
        resid_arr (np.array): Array which will receive residuals.
                          Shape is n_dir, n_tim, n_fre, n_ant, a_ant, n_cor, n_cor
        obser_arr (np.array): Array containing the observed visibilities.
                          Same shape
        model_arr (np.array): Array containing the model visibilities.
                          Same shape
        gains (np.array): Array containing the current gain estimates.
                          Shape of n_dir, n_timint, n_freint, n_ant, n_cor, n_cor
                          Where n_timint = ceil(n_tim/t_int), n_fre = ceil(n_fre/t_int)

    Returns:
        residual (np.array): Array containing the result of computing D-GMG^H.
    """
    
    gains_h = gains.transpose(0,1,2,3,5,4).conj()

    cyfull.cycompute_residual(model_arr, gains, gains_h, obser_arr, resid_arr, t_int, f_int)

    return resid_arr

def retile_array(in_arr, m1, m2, n1, n2):
    """
    Retiles a 2D array of shape m, n into shape m1, m2, n1, n2. If tiling is perfect, 
    i.e. m1*m2 = m, n1*n2 =n, then this returns a reshaped array. Otherwise, it creates a new 
    array and copies data.
    """
    
    # TODO: Investigate writing a kernel that accomplishes this and the relevant summation.

    m, n = in_arr.shape

    new_shape = (m1, m2, n1, n2)

    if (m1*m2 == m) and (n1*n2 == n):
        
        return in_arr.reshape(new_shape)
    
    else:
        
        out_arr = np.zeros(new_shape, dtype=in_arr.dtype)
        out_arr.reshape((m1*m2, n1*n2))[:m,:n] = in_arr
        
        return out_arr

# accumulates total variance per antenna/channel over the entire MS
# this is per DDID
total_deltavis2 = {}
total_deltavalid = {}

def estimate_noise (data, flags, ddid):
    """
    Given a data cube with dimensions (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) and a flag cube 
    (n_tim, n_fre, n_ant, n_ant), this function estimates the noise in the data.

    Returns tuple of noise, inverse_noise_per_antenna_channel_squared, inverse_noise_per_antenna_squared and inverse_noise_per_channel_squared.
    """

    n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape
    
    # Create a boolean flag array from the bitflags. Construct delta flags by or-ing flags in
    # channel n with flags in channel n+1.

    deltaflags = (flags!=0)
    deltaflags[:, 1:, ...] = deltaflags[:, 1:, ...] | deltaflags[:, :-1, ...]
    deltaflags[:, 0 , ...]  = deltaflags[:,   1, ...]
    
    # Create array for the squared difference between channel-adjacent visibilities.

    deltavis2 = np.zeros((n_tim, n_fre, n_ant, n_ant), np.float32)
    
    # Square the absolute value of the difference between channel-adjacent visibilities and sum 
    # over correlations. Normalize the result by n_cor*n_cor*4. The factor of 4 arises because 
    # Var<c1-c2> = Var<c1>+Var<c2> and Var<c>=Var<r>+Var<i>. Thus, the square of the abs difference
    # between two complex visibilities has contributions from _four_ noise terms.

    # TODO: When fewer than 4 correlations are provided, the normalisation needs to be different.

    deltavis2[:, 1:, ...]  = np.square(abs(data[:, 1:, ...] - data[:, :-1, ...])).sum(axis=(-2,-1))
    deltavis2[:, 1:, ...] /= n_cor*n_cor*4
    deltavis2[:, 0 , ...]  = deltavis2[:, 1, ...]
    
    # The flagged elements are zeroed; we don't have an adequate noise estimate for those channels.

    deltavis2[deltaflags] = 0

    # This flag inversion gives a count of the valid estimates in deltavis2.

    deltaflags = ~deltaflags

    # sum into n_fre, n_ant arrays
    deltavis2_chan_ant = deltavis2.sum(axis=(0, 2))  # sum, per chan, ant
    deltanum_chan_ant = deltaflags.sum(axis=(0, 2))  # number of valid points per chan, ant
    # add to global stats
    if ddid not in total_deltavis2:
        total_deltavis2[ddid] = deltavis2_chan_ant.copy()
        total_deltavalid[ddid] = deltanum_chan_ant.copy()
    else:
        total_deltavis2[ddid] += deltavis2_chan_ant
        total_deltavalid[ddid] += deltanum_chan_ant
    # now return the stddev per antenna, and per channel
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore division by 0
        noise_est = math.sqrt(deltavis2_chan_ant.sum() / deltanum_chan_ant.sum())
        inv_var_antchan =  deltavis2_chan_ant / deltanum_chan_ant
        inv_var_ant  = deltanum_chan_ant.sum(axis=0) / deltavis2_chan_ant.sum(axis=0)
        inv_var_chan = deltanum_chan_ant.sum(axis=1) / deltavis2_chan_ant.sum(axis=1)
    # antennas/channels with no data end up with NaNs here, so replace them with 0
    inv_var_antchan[~np.isfinite(inv_var_antchan)] = 0
    inv_var_ant[~np.isfinite(inv_var_ant)] = 0
    inv_var_chan[~np.isfinite(inv_var_chan)] = 0
    return noise_est, inv_var_antchan, inv_var_ant, inv_var_chan


def solve_gains(obser_arr, model_arr, flags_arr, min_delta_g=1e-6, maxiter=30,
                chi_tol=1e-5, chi_interval=5, t_int=1, f_int=1, label="", ddid=None):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array: n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the observed visibilities.
        model_arr (np.array: n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the model visibilities.
        flags_arr (np.array: n_tim, n_fre, n_ant, n_ant): int array flagging invalid points
        min_delta_g (float): Gain improvement threshold.
        maxiter (int): Maximum number of iterations allowed.
        chi_tol (float): Chi-squared improvement threshold (relative)
        chi_interval (int): Interval at which the chi-squared test is performed.

    Returns:
        gains (np.array): Array containing the final gain estimates.
    """

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model_arr.shape

    # n_tim and n_fre are the time and frequency dimensions of the data arrays.
    # n_timint and n_freint are the time and frequnecy dimensions of the gains.

    n_timint = int(math.ceil(float(n_tim)/t_int))   # Number of time intervals
    n_freint = int(math.ceil(float(n_fre)/f_int))   # Number of freq intervals
    n_tf  = n_fre*n_tim                             # Number of time-freq slots
    n_int = n_timint*n_freint                       # Number of solution intervals

    # Initialize gains to the appropriate shape with all gains set to identity. Create a copy to 
    # hold the gain of the previous iteration. 

    gain_shape = [n_dir, n_timint, n_freint, n_ant, n_cor, n_cor]
    
    gains     = np.empty(gain_shape, dtype=obser_arr.dtype)
    gains[:]  = np.eye(n_cor) 
    old_gains = gains.copy()

    # Initialize some numbers used in convergence testing.

    n_cnvgd = 0 # Number of converged solutions
    n_stall = 0 # Number of intervals with stalled chi-sq
    n_sols = float(n_dir*n_int) # Number of gains solutions
    n_vis2x2 = n_tf*n_ant*n_ant # Number of 2x2 visbilities
    iters = 0   

    # Estimates the overall noise level and the inverse variance per channel and per antenna as 
    # noise varies across the band. This is used to normalize chi^2.

    noise_est, inv_var_antchan, inv_var_ant, inv_var_chan = estimate_noise(obser_arr, flags_arr, ddid)

    print inv_var_ant

    # TODO: Check number of equations per solution interval, and deficient flag intervals.
    unflagged = (flags_arr==0)

    # (n_ant) vector containing the number of valid equations per antenna.
    # Factor of two is necessary as we have the conjugate of each equation too.

    eqs_per_antenna = 2*np.sum(unflagged, axis=(0, 1, 2))

    # (n_tim, n_fre) array containing number of valid equations per each time/freq slot.

    eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2))*n_cor*n_cor*2

    # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.
    
    eqs_per_interval = retile_array(eqs_per_tf_slot, n_timint, t_int, n_freint, f_int).sum(axis=(1,3))

    # The following determines the number of valid (unflagged) time/frequency slots and the number 
    # of valid solution intervals.

    valid_tf_slots  = eqs_per_tf_slot>0
    valid_intervals = eqs_per_interval>0
    num_valid_tf_slots  = valid_tf_slots.sum()
    num_valid_intervals = valid_intervals.sum()

    # In the event that there are no solution intervals with valid data, this will log some of the
    # flag information. This also breaks out of the function.

    if num_valid_intervals is 0:

        fstats = ""

        for flag, mask in FL.categories().iteritems():
            
            n_flag = np.sum((flags_arr & mask) != 0)/(n_cor*n_cor)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, "{}: no valid solution intervals. Flags are: \n{}".format(label, fstats)
        
        return gains
    
    # Compute chi-squared normalization factor for each solution interval.
    
    chisq_norm = np.zeros_like(eqs_per_interval, dtype=obser_arr.real.dtype)
    chisq_norm[valid_intervals] = (1./eqs_per_interval[valid_intervals])

    # Initialize a tiled residual array (tiled by whole time/freq intervals). Shapes correspond to 
    # tiled array shape and the intermediate shape from which our view of the residual is selected.
    
    tiled_shape = [n_timint, t_int, n_freint, f_int, n_ant, n_ant, n_cor, n_cor]
    inter_shape = [n_timint*t_int, n_freint*f_int, n_ant, n_ant, n_cor, n_cor]
    
    tiled_resid_arr = np.zeros(tiled_shape, obser_arr.dtype)
    resid_arr = tiled_resid_arr.reshape(inter_shape)[:n_tim,:n_fre,...]
    compute_residual(obser_arr, model_arr, resid_arr, gains, t_int, f_int)

    # Chi-squared is computed by summation over antennas, correlations and intervals. Sum over 
    # time intervals, antennas and correlations first. Normalize by per-channel variance and finally
    # sum over frequency intervals.

    chi = np.sum(np.square(np.abs(tiled_resid_arr)), axis=(1,4,5,6,7))
    chi.reshape((n_timint,n_freint*f_int))[:,:n_fre] *= inv_var_chan[np.newaxis,:]
    chi = np.sum(chi, axis=2) * chisq_norm
    init_chi = mean_chi = chi.sum() / num_valid_intervals

    # The following provides some debugging information when verbose is set to > 0. 

    if verbose > 0:

        mineqs = eqs_per_interval[valid_intervals].min()
        maxeqs = eqs_per_interval.max()
        anteqs = np.sum(eqs_per_antenna!=0)
        
        fstats = ""

        for flag, mask in FL.categories().iteritems():
            
            n_flag = np.sum((flags_arr & mask) != 0)/(n_cor*n_cor)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, ("{} Initial chisq = {:.4}, {}/{} valid intervals (min {}/max {} eqs per int)," 
                      " {}/{} valid antennas. Flags are: {}").format(   label, 
                                                                        init_chi,
                                                                        num_valid_intervals,
                                                                        n_int, 
                                                                        mineqs,
                                                                        maxeqs,
                                                                        anteqs,
                                                                        n_ant,
                                                                        fstats  )

    min_quorum = 0.99
    warned_null_gain = warned_boom_gain = False

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or 
    # stalled solutions or when the maximum number of iterations is exceeded.

    while n_cnvgd/n_sols < min_quorum and n_stall/n_int < min_quorum and iters < maxiter:
        
        iters += 1
        
        if iters % 2 == 0:
            gains = 0.5*(gains + compute_update(model_arr, obser_arr, gains, t_int, f_int))
        else:
            gains = compute_update(model_arr, obser_arr, gains, t_int, f_int)
        
        # TODO: various infs and NaNs here indicate something wrong with a solution. These should 
        # be flagged and accounted for properly in the statistics.
        
        # Compute values used in convergence tests.

        diff_g = np.sum(np.square(np.abs(old_gains - gains)), axis=(-1,-2,-3))
        norm_g = np.sum(np.square(np.abs(gains)), axis=(-1,-2,-3))
        norm_g[:,~valid_intervals] = 1      # Prevents division by zero.
        
        # Checks for unexpected null gain solutions and logs a warning.

        null_g = (norm_g==0)

        if null_g.any():
            norm_g[null_g] = 1
            if not warned_null_gain:
                print>>log, ModColor.Str("{} iteration {} WARNING: {} null gain solution(s) "
                                         "encountered".format(label, iters, null_g.sum()))
                warned_null_gain = True

        # Count converged solutions based on norm_diff_g. Flagged solutions will have a norm_diff_g
        # of 0 by construction.

        norm_diff_g = diff_g/norm_g
        n_cnvgd = np.sum(norm_diff_g <= min_delta_g**2)

        # Update old gains for subsequent convergence tests.

        old_gains = gains.copy()

        # Check residual behaviour after a number of iterations equal to chi_interval. This is 
        # expensive, so we do it as infrequently as possible.

        if (iters % chi_interval) == 0:

            old_chi, old_mean_chi = chi, mean_chi

            compute_residual(obser_arr, model_arr, resid_arr, gains, t_int, f_int)

            # TODO: Some residuals blow up and cause np.square() to overflow -- need to flag these.
            
            chi = np.sum(np.square(np.abs(tiled_resid_arr)), axis=(1, 4, 5, 6, 7))
            chi.reshape((n_timint, n_freint * f_int))[:, :n_fre] *= inv_var_chan[np.newaxis, :]
            chi = np.sum(chi, axis=2) * chisq_norm
            mean_chi = chi.sum() / num_valid_intervals

            # Check for stalled solutions - solutions for which the residual is no longer improving.

            n_stall = float(np.sum(((old_chi - chi) < chi_tol*old_chi)))

            if verbose > 1:

                delta_chi = (old_mean_chi-mean_chi)/old_mean_chi

                print>> log, ("{} iteration {} chi-sq is {:.4} delta {:.4}, max gain update {:.4}, "
                              "converged {:.2%}, stalled {:.2%}").format(   label,
                                                                            iters,
                                                                            mean_chi,
                                                                            delta_chi, 
                                                                            diff_g.max(), 
                                                                            n_cnvgd/n_sols, 
                                                                            n_stall/n_int   )

    print>>log, ("{}: {} iterations done. Converged {:.2%}, stalled {:.2%}. "
                "Chisq {:.4} -> {:.4}").format( label, 
                                                iters, 
                                                n_cnvgd/n_sols,
                                                n_stall/n_int, 
                                                init_chi, 
                                                mean_chi        )

    return gains


def apply_gains(obser_arr, gains, t_int=1, f_int=1):
    """
    Applies the inverse of the gain estimates to the observed data matrix.

    Args:
        obser_arr (np.array): Array of the observed visibilities.
        gains (np.array): Array of the gain estimates.

    Returns:
        inv_gdgh (np.array): Array containing (G^-1)D(G^-H).
    """

    g_inv = np.empty_like(gains)

    cyfull.cycompute_jhjinv(gains, g_inv) # Function can invert G.

    gh_inv = g_inv.transpose(0,1,2,3,5,4).conj()

    corr_vis = np.empty_like(obser_arr)

    cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, t_int, f_int)

    return corr_vis

def solve_and_save(obser_arr, model_arr, flags_arr, min_delta_g=1e-6, maxiter=30,
                   chi_tol=1e-6, chi_interval=5, t_int=1, f_int=1, label="", ddid=None):

    gains = solve_gains(obser_arr, model_arr, flags_arr, min_delta_g, maxiter,
                        chi_tol, chi_interval, t_int, f_int, label=label, ddid=ddid)

    corr_vis = apply_gains(obser_arr, gains, t_int, f_int)

    return gains, corr_vis

def debug():
    main(debugging=True)


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


def main(debugging=False):
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
            basename = GD["output"]["name"]
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
        global verbose
        verbose = GD["debug"]["verbose"]

        ddid, ddid_to = GD["selection"]["ddid"], GD["selection"]["ddid-to"]
        if ddid is not None and ddid_to is not None:
            ddid = ddid, ddid_to+1

        ms = ReadModelHandler(GD["data"]["ms"], GD["data"]["column"], GD["model"]["lsm"], GD["model"]["column"],
                              fid=GD["selection"]["field"], ddid=ddid,
                              precision=GD["solution"]["precision"],
                              ddes=GD["model"]["ddes"],
                              weight_column=GD["weight"]["column"])
        ms.apply_flags = bool(GD["flags"]["apply"])
        ms.bitmask = GD["flags"]["apply-bitmask"]

        print>>log, "reading MS columns"
        ms.mass_fetch()
        print>>log, "defining chunks"
        ms.define_chunk(GD["data"]["time-chunk"], GD["data"]["freq-chunk"], single_chunk_id=GD["data"]["single-chunk"])

        target = solve_and_save if GD["output"]["vis"] else solve_gains

        opts = { "min_delta_g"  : GD["solution"]["delta-g"],
                 "maxiter"      : GD["solution"]["max-iter"],
                 "chi_tol"      : GD["solution"]["delta-chi"],
                 "chi_interval" : GD["solution"]["chi-int"],
                 "t_int"        : GD["solution"]["time-int"],
                 "f_int"        : GD["solution"]["freq-int"] }


        t0 = time()

        # Debugging mode: run serially if processes is not set, or if a single chunk is specified.
        # Normal mode: use futures to run in parallel. TODO: Figure out if we can used shared memory to
        # improve performance.
        ncpu = GD["parallel"]["ncpu"]

        if debugging or ncpu <= 1 or GD["data"]["singlechunk"]:
            for obser, model, flags, weight, chunk_label in ms:

                if target is solve_and_save:
                    gains, covis = target(obser, model, flags, ddid=ms._chunk_ddid, label = chunk_label, **opts)
                    ms.arr_to_col(covis, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f])
                else:
                    gains = target(obser, model, flags, label = chunk_label, **opts)

                ms.add_to_gain_dict(gains, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f],
                                    GD["solution"]["time-int"], GD["solution"]["freq-int"])

        else:
            with cf.ProcessPoolExecutor(max_workers=ncpu) as executor:
                future_gains = { executor.submit(target, obser, model, flags, ddid=ms._chunk_ddid, label=chunk_label, **opts) :
                                 [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f]
                                 for obser, model, flags, weight, chunk_label in ms }

                for future in cf.as_completed(future_gains):

                    if target is solve_and_save:
                        gains, covis = future.result()
                        ms.arr_to_col(covis, future_gains[future])
                    else:
                        gains = future.result()

                    ms.add_to_gain_dict(gains, future_gains[future],
                                        GD["solution"]["time-int"], GD["solution"]["freq-int"])

        print>>log, ModColor.Str("Time taken: {} seconds".format(time() - t0), col="green")

        if GD["output"]["noise-plots"]:
            import pylab
            for ddid in total_deltavis2.iterkeys():
                with np.errstate(divide='ignore', invalid='ignore'):  # ignore division by 0
                    noise = np.sqrt(total_deltavis2[ddid] / total_deltavalid[ddid])
                pylab.subplot(121)
                for ant in xrange(noise.shape[1]):
                    pylab.plot(noise[:,ant])
                pylab.title("DDID {}".format(ddid))
                pylab.xlabel("channel")
                pylab.ylabel("noise")
                pylab.subplot(122)
                pylab.title("DDID {}".format(ddid))
                for chan in xrange(noise.shape[0]):
                    pylab.plot(noise[chan,:])
                pylab.xlabel("antenna")
                pylab.ylabel("noise")
                pylab.show()

        ms.write_gain_dict()

        if target is solve_and_save:
            ms.save(ms.covis, "CORRECTED_DATA")
    except Exception, exc:
        import traceback
        print>>log, ModColor.Str("Exiting with exception: {}({})\n {}".format(type(exc).__name__, exc, traceback.format_exc()))
        if enable_pdb:
            import pdb
            exc, value, tb = sys.exc_info()
            pdb.post_mortem(tb)  # more "modern"
        sys.exit(1)