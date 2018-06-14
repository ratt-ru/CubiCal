# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Source provider for reading source information from a Tigger lsm.
"""

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

        self._clusters = cluster_sources(self._sm, dde_tag)
        self._cluster_keys = self._clusters.keys()
        self._nclus = len(self._cluster_keys)

        self._target_key = 0
        self._target_cluster = self._cluster_keys[self._target_key]

        self._pnt_sources = self._clusters[self._target_cluster]["pnt"]
        self._npsrc = len(self._pnt_sources)
        self._gau_sources = self._clusters[self._target_cluster]["gau"]
        self._ngsrc = len(self._gau_sources)

    def set_direction(self, idir):
        """Sets current direction being simulated. 
        
        Args:
            idir (int):
                Direction number, from 0 to n_dir-1
        """

        self._target_key = idir
        self._target_cluster = self._cluster_keys[self._target_key]
        self._pnt_sources = self._clusters[self._target_cluster]["pnt"]
        self._npsrc = len(self._pnt_sources)
        self._gau_sources = self._clusters[self._target_cluster]["gau"]
        self._ngsrc = len(self._gau_sources)

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
            lm[ind,0], lm[ind,1] = radec_to_lm(ra, dec, self._phase_center)

        return lm

    def point_stokes(self, context):
        """ Returns a stokes parameter array to Montblanc. """

        # Get the extents of the time, baseline and chan dimension
        (lt, ut), (lp, up) = context.dim_extents('ntime', 'npsrc')

        stokes = np.empty(context.shape, context.dtype)

        for ind, source in enumerate(self._pnt_sources[lp:up]):
            stokes[ind,:,0] = source.flux.I
            stokes[ind,:,1] = source.flux.Q
            stokes[ind,:,2] = source.flux.U
            stokes[ind,:,3] = source.flux.V

        return stokes

    def point_alpha(self, context):
        """ Returns a spectral index (alpha) array to Montblanc. """

        alpha = np.empty(context.shape, context.dtype)

        (lp, up) = context.dim_extents('npsrc')

        for ind, source in enumerate(self._pnt_sources[lp:up]):
            try:
                alpha[ind] = source.spectrum.spi
            except:
                alpha[ind] = 0

        return alpha

    def point_ref_freq(self, context):
        """ Returns a reference frequency per source array to Montblanc. """
        
        pt_ref_freq = np.empty(context.shape, context.dtype)

        (lp, up) = context.dim_extents('npsrc')
        
        for ind, source in enumerate(self._pnt_sources[lp:up]):
            try:
                pt_ref_freq[ind] = source.spectrum.freq0
            except:
                pt_ref_freq[ind] = self._sm.freq0 or 0

        return pt_ref_freq


    def gaussian_lm(self, context):
        """ Returns an lm coordinate array to Montblanc. """

        lm = np.empty(context.shape, context.dtype)

        # Get the extents of the time, baseline and chan dimension
        (lg, ug) = context.dim_extents('ngsrc')

        for ind, source in enumerate(self._gau_sources[lg:ug]):

            ra, dec = source.pos.ra, source.pos.dec
            lm[ind,0], lm[ind,1] = radec_to_lm(ra, dec, self._phase_center)

        return lm

    def gaussian_stokes(self, context):
        """ Return a stokes parameter array to Montblanc """

        # Get the extents of the time, baseline and chan dimension
        (lt, ut), (lg, ug) = context.dim_extents('ntime', 'ngsrc')

        stokes = np.empty(context.shape, context.dtype)

        for ind, source in enumerate(self._gau_sources[lg:ug]):
            stokes[ind,:,0] = source.flux.I
            stokes[ind,:,1] = source.flux.Q
            stokes[ind,:,2] = source.flux.U
            stokes[ind,:,3] = source.flux.V

        return stokes


    def gaussian_alpha(self, context):
        """ Returns a spectral index (alpha) array to Montblanc """

        alpha = np.empty(context.shape, context.dtype)

        (lg, ug) = context.dim_extents('ngsrc')

        for ind, source in enumerate(self._gau_sources[lg:ug]):
            try:
                alpha[ind] = source.spectrum.spi
            except:
                alpha[ind] = 0

        return alpha


    def gaussian_shape(self, context):
        """ Returns a Gaussian shape array to Montblanc """

        shapes = np.empty(context.shape, context.dtype)

        (lg, ug) = context.dim_extents('ngsrc')

        for ind, source in enumerate(self._gau_sources[lg:ug]):
            shapes[0, ind] = source.shape.ex * np.sin(source.shape.pa)
            shapes[1, ind] = source.shape.ex * np.cos(source.shape.pa)
            shapes[2, ind] = source.shape.ey/source.shape.ex

        return shapes

    def gaussian_ref_freq(self, context):
        """ Returns a reference frequency per source array to Montblanc """

        gau_ref_freq = np.empty(context.shape, context.dtype)

        (lg, ug) = context.dim_extents('ngsrc')
        
        for ind, source in enumerate(self._gau_sources[lg:ug]):
            try:
                gau_ref_freq[ind] = source.spectrum.freq0
            except:
                gau_ref_freq[ind] = self._sm.freq0 or 0

        return gau_ref_freq

    def updated_dimensions(self):
        """ Informs Montblanc of updated dimension sizes. """

        return [('npsrc', self._npsrc),
                ('ngsrc', self._ngsrc)]

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
                if type(tagvalue) is str:
                    dde_cluster = tagvalue
                else:
                    dde_cluster = src.getTag('cluster')

        group = 'pnt' if src.typecode=='pnt' else 'gau'

        clus.setdefault(dde_cluster, dict(pnt=[], gau=[]))[group].append(src)

    return clus

def radec_to_lm(ra, dec, phase_center):
    """
    Convert right-ascension and declination to direction cosines.

    Args:
        ra (float):
            Right-ascension in radians.
        dec (float):
            Declination in radians.
        phase_center (np.ndarray):
            The coordinates of the phase center.

    Returns:
        tuple: 
            l and m coordinates.

    """

    delta_ra = ra - phase_center[...,-2]
    dec_0 = phase_center[...,-1]

    l = np.cos(dec)*np.sin(delta_ra)
    m = np.sin(dec)*np.cos(dec_0) -\
        np.cos(dec)*np.sin(dec_0)*np.cos(delta_ra)

    return l, m
