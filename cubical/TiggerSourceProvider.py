"""
Source provider for reading source information from a Tigger lsm.
"""

import logging
import numpy as np

import montblanc
import montblanc.util as mbu

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    FitsBeamSourceProvider,
    MSSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import (SinkProvider,
    MSSinkProvider)

import Tigger
import pyrap.tables as pt

class TiggerSourceProvider(SourceProvider):
    """
    Simulates sources provided by a Tigger sky model. 
    """

    def __init__(self, tile):
        """ 
        Initialises this source provider. 

        Args:
            tile (:obj:`~cubical.data_handler.Tile`):
                Tile object containing information about current data selection.
        """

        self._tile = tile
        self._handler = tile.handler
        self._sm = Tigger.load(self._handler.sm_name)
        self._phase_center = self._handler._phadir
        self._use_ddes = self._handler.use_ddes

        self._clusters = cluster_sources(self._sm, self._use_ddes)
        self._cluster_keys = self._clusters.keys()
        self._nclus = len(self._cluster_keys)

        self._target_key = 0
        self._target_cluster = self._cluster_keys[self._target_key]

        self._pnt_sources = self._clusters[self._target_cluster]["pnt"]
        self._npsrc = len(self._pnt_sources)
        self._gau_sources = self._clusters[self._target_cluster]["gau"]
        self._ngsrc = len(self._gau_sources)

    def update_target(self):
        """ Updates current target - used for direction dependent simulation. """

        if (self._target_key + 1)<self._nclus:
            self._target_key += 1
            self._target_cluster = self._cluster_keys[self._target_key]
            self._pnt_sources = self._clusters[self._target_cluster]["pnt"]
            self._npsrc = len(self._pnt_sources)
            self._gau_sources = self._clusters[self._target_cluster]["gau"]
            self._ngsrc = len(self._gau_sources)

    def name(self):
        """ Returns name of assosciated source provider. """

        return "Tigger sky model source provider"

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

def cluster_sources(sm, use_ddes):
    """
    Groups sources by shapes and tags specified in the sky model.

    Args:
        sm (:obj:`~Tigger.Models.SkyModel.SkyModel`):
            SkyModel object containing source information.
        use_ddes (bool):
            If True, take DDE and cluster tags into account. 
            Required for DD simulation.

    Returns:
        dict:
            Dictionary of grouped sources.
    """

    ddes = {'True': [], 'False': []}

    for s in sm.sources:
        if use_ddes:
            ddes['True'].append(s) if (s.getTag('dE')==True) \
                else ddes['False'].append(s)
        else:
            ddes['False'].append(s)

    clus = {}

    for s in ddes['True']:
        try:
            clus['{}'.format(s.getTag('cluster'))].append(s)
        except:
            clus['{}'.format(s.getTag('cluster'))] = [s]

    clus["die"] = ddes['False']

    for i in clus.keys():
        stype = {'pnt': [], 'gau': []}

        for s in clus[i]:
            stype['pnt'].append(s) if (s.typecode=='pnt') else stype['gau'].append(s)

        clus[i] = stype

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
