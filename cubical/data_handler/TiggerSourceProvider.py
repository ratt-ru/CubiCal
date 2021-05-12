# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Source provider for reading source information from a Tigger lsm.
"""
from six import string_types
import logging
import numpy as np

import montblanc

from montblanc.impl.rime.tensorflow.sources import SourceProvider

import Tigger
import pyrap.tables as pt


class TiggerSourceProvider(SourceProvider):
    """
    A Montblanc-compatible source provider that returns source information from a Tigger sky model.
    """
    def __init__(self, lsm, phase_center, dde_tag='dE'):
        """
        Initialises this source provider.

        Args:
            lsm (str):
                Filename containing the sky model
            phase_center (tuple):
                Observation phase centre, as a RA, Dec tuple
            dde_tag (str or None):
                If set, sources are grouped into multiple directions using the specified tag.

        """

        self.filename = lsm
        self._sm = Tigger.load(lsm)
        self._phase_center = phase_center
        self._use_ddes = bool(dde_tag)
        self._dde_tag = dde_tag

        self._freqs = None

        self._clusters = cluster_sources(self._sm, dde_tag)

        self._cluster_keys = list(self._clusters.keys())
        self._nclus = len(self._cluster_keys)

        self._target_key = 0
        self._target_cluster = self._cluster_keys[self._target_key]
        self._target_reqbeam = "beam"

        self._pnt_sources = \
            self._clusters[self._target_cluster][self._target_reqbeam]["pnt"]
        self._npsrc = len(self._pnt_sources)
        self._gau_sources = \
            self._clusters[self._target_cluster][self._target_reqbeam]["gau"]
        self._ngsrc = len(self._gau_sources)

    def set_direction(self, idir, req_beam="beam"):
        """Sets current direction being simulated.

        Args:
            idir (int):
                Direction number, from 0 to n_dir-1
        """

        self._target_key = idir
        self._target_cluster = self._cluster_keys[self._target_key]
        self._target_reqbeam = req_beam
        self._pnt_sources = \
            self._clusters[self._target_cluster][self._target_reqbeam]["pnt"]
        self._npsrc = len(self._pnt_sources)
        self._gau_sources = \
            self._clusters[self._target_cluster][self._target_reqbeam]["gau"]
        self._ngsrc = len(self._gau_sources)

    def set_frequency(self, frequency):
        """Sets simulated frequencies

        Args:
            frequency (ndarray):
                Array of frequencies associated with a DDID
        """
        self._freqs = frequency

    def name(self):
        """ Returns name of assosciated source provider. This is just the filename, in this case."""
        return self.filename

    def point_lm(self, context):
        """ Returns an lm coordinate array to Montblanc. """

        lm = np.empty(context.shape, context.dtype)

        # Print the array schema
        #Montblanc.log.info(context.array_schema.shape)
        # Print the space of iteration
        #Montblanc.log.info(context.iter_args)

        # Get the extents of the time, baseline and chan dimension
        (lp, up) = context.dim_extents('npsrc')
        assert lm.shape == (up-lp, 2)

        for ind, source in enumerate(self._pnt_sources[lp:up]):

            ra, dec = source.pos.ra, source.pos.dec
            lm[ind,0], lm[ind,1] = ra, dec #radec_to_lm(ra, dec, self._phase_center)

        return lm

    def source_spectrum(self, source, freqs):
        """
        Calculate a spectrum for the source. If source.spectrum attribute is missing,
        returns flat spectrum

        Args:
            source (:obj:`~Tigger.Models.SkyModel.Source`):
                Tigger source
            freq (`numpy.ndarray`):
                array of frequencies of shape `(chan,)`

        Returns:
            `numpy.ndarray`:
                spectrum of shape `(chan,)`
        """
        if hasattr(source, 'spectrum') and source.spectrum is not None:
            rf = getattr(source.spectrum, 'freq0', 1e+9)
            alpha = source.spectrum.spi
            frf = freqs / rf
            if not isinstance(alpha, (list, tuple)):
                alpha = [alpha]
            logfr = np.log10(frf)
            spectrum = frf ** sum([a * np.power(logfr, n) for n, a in enumerate(alpha)])

            ## broadcast into the time dimension.
            return spectrum
        else:
            return np.ones_like(freqs)

    def point_stokes(self, context):
        """ Returns a stokes parameter array to Montblanc. """

        # Get the extents of the time, baseline and chan dimension
        (lp, up), (lt, ut), (lc, uc) = context.dim_extents('npsrc',
                                                           'ntime',
                                                           'nchan')
        # (npsrc, ntime, nchan, 4)
        stokes = np.empty(context.shape, context.dtype)

        f = self._freqs[lc:uc].ravel()

        for ind, source in enumerate(self._pnt_sources[lp:up]):
            spectrum = self.source_spectrum(source, f)[None, :]

            # Multiply flux into the spectrum,
            # broadcasting into the time dimension
            for iS, S in enumerate('IQUV'):
                stokes[ind, :, :, iS] = getattr(source.flux, S, 0.)*spectrum

        return stokes

    def gaussian_lm(self, context):
        """ Returns an lm coordinate array to Montblanc. """

        lm = np.empty(context.shape, context.dtype)

        # Get the extents of the time, baseline and chan dimension
        (lg, ug) = context.dim_extents('ngsrc')

        for ind, source in enumerate(self._gau_sources[lg:ug]):

            ra, dec = source.pos.ra, source.pos.dec
            lm[ind, 0], lm[ind, 1] = ra, dec #radec_to_lm(ra, dec, self._phase_center)

        return lm

    def gaussian_stokes(self, context):
        """ Return a stokes parameter array to Montblanc """

        # Get the extents of the source, time and channel dims
        (lg, ug), (lt, ut), (lc, uc) = context.dim_extents('ngsrc',
                                                           'ntime',
                                                           'nchan')
        # (npsrc, ntime, nchan, 4)
        stokes = np.empty(context.shape, context.dtype)

        f = self._freqs[lc:uc].ravel()

        for ind, source in enumerate(self._gau_sources[lg:ug]):
            spectrum = self.source_spectrum(source, f)[None, :]

            # Multiply flux into the spectrum,
            # broadcasting into the time dimension
            for iS, S in enumerate('IQUV'):
                stokes[ind, :, :, iS] = getattr(source.flux, S, 0.)*spectrum

        return stokes

    def gaussian_shape(self, context):
        """ Returns a Gaussian shape array to Montblanc """

        shapes = np.empty(context.shape, context.dtype)

        (lg, ug) = context.dim_extents('ngsrc')

        for ind, source in enumerate(self._gau_sources[lg:ug]):
            shapes[0, ind] = source.shape.ex * np.sin(source.shape.pa)
            shapes[1, ind] = source.shape.ex * np.cos(source.shape.pa)
            shapes[2, ind] = source.shape.ey/source.shape.ex

        return shapes

    def updated_dimensions(self):
        """ Informs Montblanc of updated dimension sizes. """

        return [('npsrc', self._npsrc),
                ('ngsrc', self._ngsrc)]

    def phase_centre(self, context):
        """ Sets the MB phase direction """
        radec = np.array([self._phase_center[...,-2],
                          self._phase_center[...,-1]], context.dtype)
        return radec

def cluster_sources(sm, dde_tag):
    """
    Groups sources by shape (point/gaussian) and, optionally, direction.

    Args:
        sm (:obj:`~Tigger.Models.SkyModel.SkyModel`):
            SkyModel object containing source information.
        dde_tag (str or None):
            If given, then also group by direction using the given tag.

    Returns:
        dict:
            Dictionary of grouped sources.
    """

    clus = {}

    # cluster sources by value of dde_tag, or if dde_tag is True but not a string,
    # then by their 'cluster' attribute. Within a cluster, split into point sources
    # and Gaussians
    for src in sm.sources:
        dde_cluster = "die"
        if dde_tag:
            tagvalue = src.getTag(dde_tag)
            if tagvalue:
                if isinstance(tagvalue, string_types):
                    dde_cluster = tagvalue
                else:
                    dde_cluster = src.getTag('cluster')

        req_beam = 'nobeam' if src.getTag('nobeam') else 'beam'
        src_type = 'pnt' if src.typecode=='pnt' else 'gau'

        clus.setdefault(dde_cluster, dict(beam=dict(pnt=[], gau=[]),
                                          nobeam=dict(pnt=[], gau=[])))
        clus[dde_cluster][req_beam][src_type].append(src)

    return clus

# def radec_to_lm(ra, dec, phase_center):
#     """
#     DEPRICATED: Montblanc now implements WCS conversions internally

#     Convert right-ascension and declination to direction cosines.

#     Args:
#         ra (float):
#             Right-ascension in radians.
#         dec (float):
#             Declination in radians.
#         phase_center (np.ndarray):
#             The coordinates of the phase center.

#     Returns:
#         tuple:
#             l and m coordinates.

#     """

#     delta_ra = ra - phase_center[...,-2]
#     dec_0 = phase_center[...,-1]

#     l = np.cos(dec)*np.sin(delta_ra)
#     m = np.sin(dec)*np.cos(dec_0) -\
#         np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)

#     return l, m
