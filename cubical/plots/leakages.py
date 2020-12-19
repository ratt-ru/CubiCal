#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import matplotlib.patches as mpatches
from pylab import *
from cubical.plots import gainsols

def plot_leakages_cc(D, FS=None, TS=None, ANTS=slice(None), refant=None,
                     plot_diag='ap', plot_offdiag='ri', figtitle=None):
    """Plots leakages from a CubiCal database"""
    sols, have_offdiag, is_complex = gainsols.prepare_sols_dict(D, FS, TS)
    if not have_offdiag:
        plot_offdiag = ''
    if not is_complex:
        plot_diag = plot_diag and 'r'
        plot_offdiag = plot_offdiag and 'r'

    # renormalize
    sols = OrderedDict([(ant, (t, f, d00, d01/d00, d10/d11, d11))
                        for ant, (t, f, d00, d01, d10, d11) in sols.items()])

    # apply new reference antenna
    if refant is not None:
        d01ref = sols[refant][3]
        sols = OrderedDict([(ant, (t, f, d00, d01 - d01ref, d10 + np.conj(d01ref), d11))
                            for ant, (t, f, d00, d01, d10, d11) in sols.items()])

    return gainsols.plot_bandpass(sols, plot_diag=plot_diag, plot_offdiag=plot_offdiag,
                                  gaintype=("Bandpass", "Leakage"), figtitle=figtitle)


def plot_leakages_aips(Daips, FS=None, ANTS=slice(None),
                       plot_diag=None, plot_offdiag='ri', diff=False, figtitle=None):
    """
    Plots AIPS leakage solutions
    """
    Da, all_ants, all_freqs = Daips
    FS = gainsols.get_freq_slice(FS, all_freqs)
    sols = OrderedDict()
    # get valid D solutions
    if isinstance(ANTS, (list, tuple)):
        ANTS = [(ant, all_ants[ant]) for ant in ANTS]
    else:
        ANTS = enumerate(all_ants[ANTS])
    time = np.array([0.])
    freq = all_freqs[FS]
    for iant,ant in ANTS:
        d00 = Da[:, FS, iant, 0, 0]
        d01 = Da[:, FS, iant, 0, 1]
        d10 = Da[:, FS, iant, 1, 0]
        d11 = Da[:, FS, iant, 1, 1]
        sols[ant] = time,freq,d00,d01,d10,d11

    return gainsols.plot_bandpass(sols, plot_diag=plot_diag, plot_offdiag=plot_offdiag,
                         gaintype=("Diff bandpass", "Diff leakage") if diff else ("Bandpass", "Leakage"),
                         figtitle=figtitle)

def read_aips_leakages(filename):
    """
    Reads AIPS leakage solutions from text file.

    Returns:
        leakage_array [(1,NFREQ,NANT,2,2) complex], antenna_names [(NANT) str], frequencies [(NFREQ) float]
    """
    import re, cmath
    from collections import OrderedDict
    allfreqs = set()
    aips_rel_leak = OrderedDict()
    antenna = antdict = None
    for iline, line in enumerate(open(filename).readlines()):
        match = re.match("\s*Antenna #\s+(\d+)\s+ name: (\w+)", line)
        if match:
            antenna, antname = match.groups()
            # print("Found antenna {} ({})".format(antenna, antname))
            antdict = aips_rel_leak.setdefault(antenna, dict(R=OrderedDict(), L=OrderedDict()))
            continue
        fields = line.strip().split()
        if len(fields) == 7:
            try:
                pol = fields[2]
                freq = float(fields[3]) * 1e+6
                amp = float(fields[5])
                phase = float(fields[6])
            except:
                continue
            if antdict is None:
                print("line {}: leakage entry before active antenna entry".format(iline + 1))
                continue
            antdict[pol][freq] = d = amp * cmath.exp(1j * phase * cmath.pi / 180)
            allfreqs.add(freq)
            # print("  {} {} MHz: {}".format(pol, freq, d))
    # filter out missing antennas
    missing = []
    for antenna, antdict in aips_rel_leak.items():
        if (np.array([antdict['R'].values(), antdict['L'].values()]) == 0).all():
            missing.append(antenna)
    if missing:
        print("  missing antennas: {}".format(" ".join(missing)))
    for m in missing:
        del aips_rel_leak[m]
    freqs = np.array(sorted(allfreqs))
    freq_index = {f: i for i, f in enumerate(freqs)}
    ants = list(aips_rel_leak.keys())
    ant_index = {a: i for i, a in enumerate(ants)}
    leakage = np.zeros((1, len(allfreqs), len(ants), 2, 2), np.complex64)
    print("  {} antennas: {}".format(len(ants), " ".join(ants)))
    print("  freqs {} to {} MHz".format(*(freqs[[0, -1]] * 1e-6)))

    for antenna, antdict in aips_rel_leak.items():
        iant = ant_index[antenna]
        for icorr, corr in enumerate('RL'):
            corrdict = antdict[corr]
            ifreqs = [freq_index[f] for f in corrdict.keys()]
            leakage[0, ifreqs, iant, icorr, 1 - icorr] = np.array(corrdict.values())
    return leakage, ants, freqs

def apply_ref_ant(leak, refant, ant_index):
    iref = ant_index.get(refant)
    if iref is None:
        print("unknown reference antenna '{}'".format(refant))
    else:
        refleak = leak[:, :, iref, 0, 1].copy()
        leak[:, :, :, 0, 1] -= refleak[..., np.newaxis]
        leak[:, :, :, 1, 0] += np.conj(refleak)[..., np.newaxis]

def subtract_leakages(leak, antennas, leak0, ant0_index):
    diffleak = leak.copy()
    for iant, ant in enumerate(antennas):
        iant0 = ant0_index.get(ant)
        if iant0 is None:
            print("antenna {} not in second file, difference will be zero'ed".format(ant))
            diffleak[:, :, iant, :, :] = 0
        else:
            diffleak[:, :, iant, :, :] -= leak0[:, :, iant0, :, :]
    return diffleak

def read_cubical_leakages(filename):
    return gainsols.read_cubical_gains(filename, "D")
