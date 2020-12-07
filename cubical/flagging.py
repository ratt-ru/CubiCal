# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles the flagging of data. 
"""

# This is to keep matplotlib from falling over when no DISPLAY is set (which it otherwise does, 
# even if one is only trying to save figures to .png.
from __future__ import print_function
from builtins import range
from past.builtins import cmp
from functools import cmp_to_key
import numpy as np
import re

from cubical.tools import logger, ModColor
import cubical.plots as plots
log = logger.getLogger("flagging")
from collections import OrderedDict

class FL(object):
    """ Namespace for flag bits. """

    dtype = np.uint16   # dtype used for flag arrays

    PRIOR    = dtype(1<<0)    # prior flags (i.e. from MS)
    MISSING  = dtype(1<<1)    # missing data
    INVALID  = dtype(1<<2)    # invalid data (zero, inf, nan)
    ILLCOND  = dtype(1<<3)    # solution ill conditioned - bad inverse
    DIVERGE  = dtype(1<<4)    # divergence
    NOSOL    = dtype(1<<5)    # missing solution (i.e. in load-from or xfer-from)
    GOOB     = dtype(1<<6)    # gain solution out of bounds
    BOOM     = dtype(1<<7)    # gain solution exploded (i.e. went to inf/nan)
    GNULL    = dtype(1<<8)    # gain solution gone to zero.
    LOWSNR   = dtype(1<<9)    # prior SNR too low for gain solution
    GVAR     = dtype(1<<10)   # posterior variance too high for gain solution
    INVMODEL = dtype(1<<11)   # invalid model (zero, inf, nan)
    INVWGHT  = dtype(1<<12)   # invalid weight (inf or nan)
    NULLWGHT = dtype(1<<13)   # null weight

    MAD      = dtype(1<<14)   # residual exceeds MAD-based threshold
    SKIPSOL  = dtype(1<<15)   # omit this data point from the solver (not a true flag!)

    @staticmethod
    def categories():
        """ Returns dict of all possible flag categories. """

        return OrderedDict([(attr, value) for attr, value in FL.__dict__.items()
                            if attr[0] != "_" and type(value) is FL.dtype])

class Flagsets (object):
    """ Manage an measurement set's flagsets. Pasted from Cattery.Meow.MSUtils. """

    def __init__ (self,ms):
        """
        Initialises a Flagsets object for the measurement set.

        Args:
            ms (:obj:`~casacore.tables.table.table`):
                A table object belonging to the measurement set.
        """

        self.ms = ms
        if not 'BITFLAG' in ms.colnames():
            self.order = None
            self.bits = {}
        else:
            kws = ms.colkeywordnames('BITFLAG')
            self.bits  = {}
            # scan FLAGSET_xxx keywords and populate name->bitmask mappings
            for kw in kws:
                match = re.match('^FLAGSET_(.*)$',kw)
                if match:
                    name = match.group(1)
                    bit = ms.getcolkeyword('BITFLAG',kw)
                    if isinstance(bit,int):
                        self.bits[name] = bit
                    else:
                        print("Warning: unexpected type (%s) for %s keyword of BITFLAG column," \
                                " ignoring"%(type(bit),kw))
            # have we found any FLAGSET_ specs?
            if self.bits:
                order = 'FLAGSETS' in kws and ms.getcolkeyword('BITFLAG','FLAGSETS')
                if isinstance(order,str):
                    order = order.split(',')
                else:
                    print("Warning: unexpected type (%s) for FLAGSETS keyword of BITFLAG column," \
                                " ignoring"%type(order))
                    order = []
                # form up "natural" order by comparing bitmasks
                bitwise_order = list(self.bits.keys())
                bitwise_order.sort(key=cmp_to_key(lambda a,b:cmp(self.bits[a],self.bits[b])))
                # if an order is specified, make sure it is actually valid,
                # and add any elements from bitwise_order that are not present
                self.order = [ fs for fs in order if fs in self.bits ] + \
                             [ fs for fs in bitwise_order if fs not in order ]
                # if order was fixed compared to what was in MS, write back to MS
                if ms.iswritable() and self.order != order:
                    ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order))
                    ms.flush()
            # else if no flagsets found, try the old-style NAMES keyword
            elif 'NAMES' in kws:
                names = ms.getcolkeyword('BITFLAG','NAMES')
                if isinstance(names,(list,tuple)):
                    self.order = list(map(str,names))
                    bit = 1
                    for name in self.order:
                        self.bits[name] = bit
                        bit <<= 1
                    if ms.iswritable():
                        ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order))
                        for name,bit in self.bits.items():
                            ms._putkeyword('BITFLAG','FLAGSET_%s'%name,-1,False,bit)
                        ms.flush()
            else:
                self.order = []

    def names (self):
        """
        Convenience function for determining active flagsets.

        Returns:
            list or None:
                A list of flagset names, in the order in which they were created or None if BITFLAG 
                column is missing (so flagsets are unavailable.)
        """

        return self.order

    def flagmask (self,name,create=False):
        """
        Flagmask corresponding to named flagset.

        Args:
            name (str):
                Name of flagset.
            create (bool, optional): 
                If True and flagset is missing, creates named flagset, else raises exception.
    
        Raises:
            TypeError:
                If the MS does not contain a BITFLAG column.
            ValueError:
                If the named flagset is not found and create is False. 
            ValueError:
                If there are too many flagsets to create a new one.
        """

        # Cludge for python2/3 interoperability.
        name = str(name)

        # lookup flagbit, return if found
        if self.order is None:
            raise TypeError("MS does not contain a BITFLAG column. Please run the addbitflagcol" \
                                                                        " utility on this MS.")
        bit = self.bits.get(name,None)
        if bit is not None:
            return bit
        # raise exception if not allowed to create a new one
        if not create:
            raise ValueError("Flagset '%s' not found"%name)
        # find empty bit
        for bitnum in range(32):
            bit = 1<<bitnum
            if bit not in list(self.bits.values()):
                self.order.append(name)
                self.bits[name] = bit
                self.ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order))
                self.ms._putkeyword('BITFLAG','FLAGSET_%s'%name,-1,False,bit)
                self.ms.flush()
                return bit
        # no free bit found, bummer
        raise ValueError("Too many flagsets in MS, cannot create another one")

    def remove_flagset (self, *fsnames):
        """
        Removes the named flagset(s). 

        Args:
            fsnames (tuple):
                Names of flagsets to be removed.

        Returns:
            int:
                Flagmask corresponding to the removed flagsets.
        """
        
        # lookup all flagsets, raise error if any not found
        if self.bits is None:
            raise TypeError("MS does not contain a BITFLAG column, cannot use flagsets")
        removing = []
        for fs in fsnames:
            bit = self.bits.get(fs,None)
            if bit is None:
                raise ValueError("Flagset '%s' not found"%fs)
            removing.append((fs,bit))
        if not removing:
            return
        # remove items, form up mask of bitflags to be cleared
        mask = 0
        for name,bit in removing:
            mask |= bit
            del self.bits[name]
            del self.order[self.order.index(name)]
            self.ms.removecolkeyword('BITFLAG','FLAGSET_%s'%name)
        # write new list of bitflags
        self.ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order))
        self.ms.flush()

        return mask

def flag_chisq (st, GD, basename, nddid):
    """
    Flags timeslots and channels based on accumulated chi-squared statistics.

    Args:
        st (:obj:`~cubical.statistics.SolverStats`):
            Object containing solver statistics.
        GD (dict):
            Dictionary of global options.
        basename (str):
            Base name for output plots.
        nddid (int):
            Number of data descriptor identifiers.

    Returns:
        np.ndarray:
            Flag cube of shape (n_times, n_ddids, n_chans).
    """

    chi2 = np.ma.masked_array(st.timechan.chi2, st.timechan.chi2==0)
    total = (~chi2.mask).sum()
    if not total:
        print(ModColor.Str("no valid solutions anywhere: skipping post-solution flagging."), file=log)
        return None

    chi2n = st.timechan.chi2n
    chi2n = np.ma.masked_array(chi2n, chi2n == 0)

    median = np.ma.median(chi2)
    median_np = np.ma.median(chi2n)
    print("median chi2 value is {:.3} from {} valid t/f slots".format(median, total), file=log)
    print("median count per slot is {}".format(median_np), file=log)

    chi_median_thresh = GD["postmortem"]["tf-chisq-median"]
    np_median_thresh  = GD["postmortem"]["tf-np-median"]
    time_density      = GD["postmortem"]["time-density"]
    chan_density      = GD["postmortem"]["chan-density"]
    ddid_density      = GD["postmortem"]["ddid-density"]

    make_plots = GD["out"]["plots"]
    show_plots = make_plots == "show"

    if make_plots:
        import pylab
        pylab.figure(figsize=(32, 10))
        pylab.subplot(161)
        if chi2.count():
            pylab.imshow(chi2, vmin=0, vmax=5 * median)
        pylab.title("$\chi^2$")
        pylab.colorbar()
        pylab.subplot(162)
        if chi2n.count():
            pylab.imshow(chi2n)
        pylab.title("counts")
        pylab.colorbar()

    flag = (chi2 > chi_median_thresh * median)
    chi2[flag] = np.ma.masked
    nflag = flag.sum()
    print("{} slots ({:.2%}) flagged on chi2 > {}*median".format(nflag, nflag/float(total), 
                                                                                chi_median_thresh), file=log)

    if make_plots:
        pylab.subplot(163)
        if chi2.count():
            pylab.imshow(chi2)
        pylab.title("$\chi^2$ median flagged")
        pylab.colorbar()

    flag2 = (chi2n < np_median_thresh * median_np)
    n_new = (flag2&~flag).sum()
    print("{} more slots ({:.2%}) flagged on counts < {}*median".format(n_new, 
                                                            n_new/float(total), np_median_thresh), file=log)
    flag |= flag2

    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(164)
        if chi2.count():
            pylab.imshow(chi2)
        pylab.title("counts flagged")
        pylab.colorbar()

    nt, nf = flag.shape

    # flag channels with overdense flagging
    freqcount = flag.sum(axis=0)
    freqflags = freqcount > nt * chan_density
    n_new = (freqflags&~(freqcount==nt)).sum()
    print("{} more channels flagged on density > {}".format(n_new, chan_density), file=log)

    # flag timeslots with overdense flagging
    timecount = flag.sum(axis=1)
    timeflags = timecount > nf * time_density
    n_new = (timeflags&~(timecount==nf)).sum()
    print("{} more timeslots flagged on density > {}".format(n_new, time_density), file=log)

    flag = flag | freqflags[np.newaxis,:] | timeflags[:,np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(165)
        if chi2.count():
            pylab.imshow(chi2)
        pylab.title("overdense flagged")
        pylab.colorbar()

    # reshape flag array into time, ddid, channel
    flag3 = flag.reshape((nt, nddid, nf // nddid))

    # flag entire DDIDs with overdense flagging
    maxcount = nt*nf/nddid
    ddidcounts = flag3.sum(axis=(0, 2))
    ddidflags = ddidcounts > maxcount * ddid_density
    n_new = (ddidflags&~(ddidcounts==maxcount)).sum()
    print("{} more ddids flagged on density > {}".format(n_new, ddid_density), file=log)

    flag3 |= ddidflags[np.newaxis, :, np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(166)
        pylab.title("overdense DDID")
        if chi2.count():
            pylab.imshow(chi2)
        pylab.colorbar()
        filename = basename+".chiflag.png"
        pylab.savefig(filename, DPI=plots.DPI)
        print("saved chi-sq flagging plot to "+filename, file=log)
        if show_plots:
            pylab.show()

    return flag3
