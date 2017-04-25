# this is to keep matplotlib from falling over when no DISPLAY is set (which it otherwise does, even if one is only
# trying to save figures to .png...)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pyrap.tables as pt
import re

from cubecal.tools import logger
log = logger.getLogger("flagging")
import plots
from collections import OrderedDict

class FL(object):
    """Namespace for flag bits"""
    PRIOR    = 1       # prior flags (i.e. from MS)
    MISSING  = 1<<1    # missing data
    INVALID  = 1<<2    # invalid data or model (inf, nan)
    NOCONV   = 1<<4    # no convergence
    CHISQ    = 1<<5    # excessive chisq
    GOOB     = 1<<6    # gain solution out of bounds
    BOOM     = 1<<7    # gain solution exploded (i.e. went to inf/nan)
    GNULL    = 1<<8    # gain solution gone to zero

    @staticmethod
    def categories():
        """Returns dict of all flag categories defined above"""
        return OrderedDict([(attr, value) for attr, value in FL.__dict__.iteritems() if attr[0] != "_" and type(value) is int])

class Flagsets (object):
  """Flagsets implements a class to manage an MS's flagsets. Pasted from Cattery.Meow.MSUtils"""
  def __init__ (self,ms):
    self.msname = ms.name()
    if not 'BITFLAG' in ms.colnames():
      self.order = None;
      self.bits = {};
    else:
      kws = ms.colkeywordnames('BITFLAG');
      self.bits  = {};
      # scan FLAGSET_xxx keywords and populate name->bitmask mappings
      for kw in kws:
        match = re.match('^FLAGSET_(.*)$',kw);
        if match:
          name = match.group(1);
          bit = ms.getcolkeyword('BITFLAG',kw);
          if isinstance(bit,int):
            self.bits[name] = bit;
          else:
            print "Warning: unexpected type (%s) for %s keyword of BITFLAG column, ignoring"%(type(order),kw);
      # have we found any FLAGSET_ specs?
      if self.bits:
        order = 'FLAGSETS' in kws and ms.getcolkeyword('BITFLAG','FLAGSETS');
        if isinstance(order,str):
          order = order.split(',');
        else:
          print "Warning: unexpected type (%s) for FLAGSETS keyword of BITFLAG column, ignoring"%type(order);
          order = [];
        # form up "natural" order by comparing bitmasks
        bitwise_order = list(self.bits.iterkeys());
        bitwise_order.sort(lambda a,b:cmp(self.bits[a],self.bits[b]));
        # if an order is specified, make sure it is actually valid,
        # and add any elements from bitwise_order that are not present
        self.order = [ fs for fs in order if fs in self.bits ] + \
                     [ fs for fs in bitwise_order if fs not in order ];
        # if order was fixed compared to what was in MS, write back to MS
        if ms.iswritable() and self.order != order:
          ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order));
          ms.flush();
      # else if no flagsets found, try the old-style NAMES keyword
      elif 'NAMES' in kws:
        names = ms.getcolkeyword('BITFLAG','NAMES');
        if isinstance(names,(list,tuple)):
          self.order = map(str,names);
          bit = 1;
          for name in self.order:
            self.bits[name] = bit;
            bit <<= 1;
          if ms.iswritable():
            ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order));
            for name,bit in self.bits.iteritems():
              ms._putkeyword('BITFLAG','FLAGSET_%s'%name,-1,False,bit);
            ms.flush();
      else:
        self.order = [];

  def names (self):
    """Returns a list of flagset names, in the order in which they were
    created. Returns None if BITFLAG column is missing (so flagsets are
    not available.)""";
    return self.order;

  def flagmask (self,name,create=False):
    """Returns flagmask corresponding to named flagset. If flagset does not exist:
      * if create is True, creates a new one
      * if create is False, raises exception
    """;
    # lookup flagbit, return if found
    if self.order is None:
      raise TypeError,"MS does not contain a BITFLAG column. Please run the addbitflagcol utility on this MS.""";
    bit = self.bits.get(name,None);
    if bit is not None:
      return bit;
    # raise exception if not allowed to create a new one
    if not create:
      raise ValueError,"Flagset '%s' not found"%name;
    # find empty bit
    for bitnum in range(32):
      bit = 1<<bitnum;
      if bit not in self.bits.values():
        self.order.append(name);
        self.bits[name] = bit;
        ms = pt.table(self.msname,readonly=False,ack=False);
        ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order));
        ms._putkeyword('BITFLAG','FLAGSET_%s'%name,-1,False,bit);
        ms.flush();
        return bit;
    # no free bit found, bummer
    raise ValueError,"Too many flagsets in MS, cannot create another one";

  def remove_flagset (self,*fsnames):
    """Removes the named flagset(s). Returns flagmask corresponding to the removed
    flagsets.""";
    # lookup all flagsets, raise error if any not found
    if self.bits is None:
      raise TypeError,"MS does not contain a BITFLAG column, cannot use flagsets""";
    removing = [];
    for fs in fsnames:
      bit = self.bits.get(fs,None);
      if bit is None:
        raise ValueError,"Flagset '%s' not found"%fs;
      removing.append((fs,bit));
    if not removing:
      return;
    # remove items, form up mask of bitflags to be cleared
    ms = pt.table(self.msname,readonly=False, ack=False);
    mask = 0;
    for name,bit in removing:
      mask |= bit;
      del self.bits[name];
      del self.order[self.order.index(name)];
      ms.removecolkeyword('BITFLAG','FLAGSET_%s'%name);
    # write new list of bitflags
    ms._putkeyword('BITFLAG','FLAGSETS',-1,False,','.join(self.order));
    ms.flush();
    return mask;





def flag_chisq (st, GD, basename, nddid):
    """
    Flags timeslots and channels based on accumulated statistics

    Returns flag cube of n_times, n_ddids, n_chans

    """
    chi2 = np.ma.masked_array(st.timechan.chi2, st.timechan.chi2==0)
    total = (~chi2.mask).sum()
    chi2n = st.timechan.chi2n
    chi2n = np.ma.masked_array(chi2n, chi2n == 0)

    median = np.ma.median(chi2)
    median_np = np.ma.median(chi2n)
    print>>log, "median chi2 value is {:.3} from {} valid t/f slots".format(median, total)
    print>>log, "median count per slot is {}".format(median_np)

    chi_median_thresh = GD["flags"]["tf-chisq-median"]
    np_median_thresh  = GD["flags"]["tf-np-median"]
    time_density      = GD["flags"]["time-density"]
    chan_density      = GD["flags"]["chan-density"]
    ddid_density      = GD["flags"]["ddid-density"]

    make_plots = GD["out"]["plots"]
    show_plots = GD["out"]["plots-show"]


    if make_plots:
        import pylab
        pylab.figure(figsize=(32, 10))
        pylab.subplot(161)
        pylab.imshow(chi2, vmin=0, vmax=5 * median)
        pylab.title("$\chi^2$")
        pylab.colorbar()
        pylab.subplot(162)
        pylab.imshow(chi2n)
        pylab.title("counts")
        pylab.colorbar()

    flag = (chi2 > chi_median_thresh * median)
    chi2[flag] = np.ma.masked
    nflag = flag.sum()
    print>>log, "{} slots ({:.2%}) flagged on chi2 > {}*median".format(nflag, nflag/float(total), chi_median_thresh)

    if make_plots:
        pylab.subplot(163)
        pylab.imshow(chi2)
        pylab.title("$\chi^2$ median flagged")
        pylab.colorbar()

    flag2 = (chi2n < np_median_thresh * median_np)
    n_new = (flag2&~flag).sum()
    print>>log, "{} more slots ({:.2%}) flagged on counts < {}*median".format(n_new, n_new/float(total), np_median_thresh)
    flag |= flag2

    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(164)
        pylab.imshow(chi2)
        pylab.title("counts flagged")
        pylab.colorbar()

    nt, nf = flag.shape

    # flag channels with overdense flagging
    freqcount = flag.sum(axis=0)
    freqflags = freqcount > nt * chan_density
    n_new = (freqflags&~(freqcount==nt)).sum()
    print>>log, "{} more channels flagged on density > {}".format(n_new, chan_density)

    # flag timeslots with overdense flagging
    timecount = flag.sum(axis=1)
    timeflags = timecount > nf * time_density
    n_new = (timeflags&~(timecount==nf)).sum()
    print>>log, "{} more timeslots flagged on density > {}".format(n_new, time_density)

    flag = flag | freqflags[np.newaxis,:] | timeflags[:,np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(165)
        pylab.imshow(chi2)
        pylab.title("overdense flagged")
        pylab.colorbar()

    # reshape flag array into time, ddid, channel
    flag3 = flag.reshape((nt, nddid, nf / nddid))

    # flag entire DDIDs with overdense flagging
    maxcount = nt*nf/nddid
    ddidcounts = flag3.sum(axis=(0, 2))
    ddidflags = ddidcounts > maxcount * ddid_density
    n_new = (ddidflags&~(ddidcounts==maxcount)).sum()
    print>>log, "{} more ddids flagged on density > {}".format(n_new, ddid_density)

    flag3 |= ddidflags[np.newaxis, :, np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(166)
        pylab.imshow(chi2)
        pylab.title("overdense DDID")
        pylab.colorbar()
        filename = basename+".chiflag.png"
        pylab.savefig(filename, DPI=plots.DPI)
        print>> log, "saved chi-sq flagging plot to "+filename
        if show_plots:
            pylab.show()

    return flag3