import pyrap.quanta as pq
import pyrap.measures
pm = pyrap.measures.measures()
import copy
import numpy as np
import datetime as dt

from cubical.tools import logger, ModColor
log = logger.getLogger("parallactic_machine")

class parallactic_machine(object):
    def __init__(self, 
                 observer_names, 
                 ECEF_positions, 
                 epoch='J2000', 
                 feed_basis='linear', 
                 enable_rotation=True, 
                 enable_derotation=True,
                 field_centre=(0,-90)):
        """
        Machine to rotate and derotate visibilities around 3rd axis of observer's local
        coordiante frame
        Args:
        @observer_names: list of antenna names
        @ECEF_positions: as provided in MS::ANTENNA subtable
        @epoch: coordinate system epoch, usually J2000
        @feed_basis: linear or circular feeds
        @enable_rotation: switches on rotation
        @enable_derotation: switches on derotation
        @field_centre: initial field centre, SCP by default
        """
        if not len(observer_names) == ECEF_positions.shape[0]:
            raise ValueError("Must provide a coordinate for every antenna")
        if not ECEF_positions.shape[1] == 3:
            raise ValueError("ECEF coordinates must be 3 dimensional")

        self.__observer_names = copy.deepcopy(observer_names)
        log.info("Initializing new parallactic angle machine for the following ECEF positions:")
        for s in ["%s\t\t%.2f\t%.2f\t%.2f" % (aname, X, Y, Z) 
                  for aname, (X, Y, Z) in zip(self.__observer_names,
                                              ECEF_positions)]:
            log.info("\t" + s)
        self.__observer_positions = [pm.position('itrf',
                                                 *(pq.quantity(x, 'm') for x in pos))
                                     for pos in ECEF_positions]
        
        self.__epoch = epoch
        log.info("Conversion epoch is %s" % self.__epoch)

        # initialize field centre to SCP
        self.__zenith_azel = pm.direction("AZEL", *(pq.quantity(fi, 'deg') for fi in (0, 90)))
        self.__observer_zenithal_angle = None
        self.__field_centre = None
        self.field_centre = field_centre

        self.feed_basis = feed_basis
        self.__enable_rotation = None
        self.__enable_derotation = None
        self.enable_rotation = enable_rotation
        self.enable_derotation = enable_derotation

    def __mjd2dt(self, utc_timestamp):
        """
        Converts array of UTC timestamps to list of datetime objects for human readable printing
        """
        return [dt.datetime.utcfromtimestamp(pq.quantity(t, "s").to_unix_time()) for t in utc_timestamp]

    def parallactic_angle(self, utc_timestamp):
        """
        Computes the parallactic angle based on the observers' zenithal angle for every timestamp
        Args:
        utc_timestamp: ndarray of utc_timestamps in seconds
        """
        dt_start = self.__mjd2dt([np.min(utc_timestamp)])[0].strftime('%Y/%m/%d %H:%M:%S')
        dt_end = self.__mjd2dt([np.max(utc_timestamp)])[0].strftime('%Y/%m/%d %H:%M:%S')
        log.info("Computing parallactic angles for times between %s and %s UTC" % (dt_start, dt_end))
        
        unique_times = np.unique(utc_timestamp)
        unique_pa = np.asarray([
            pm.do_frame(pm.epoch("UTC", pq.quantity(t, 's')))
            and
            [
                pm.do_frame(rp)
                and
                pm.posangle(self.field_centre[rpi], self.__zenith_azel).get_value("rad")
                for rpi, rp in enumerate(self.__observer_positions)
            ]
            for t in unique_times])
        
        ntime = utc_timestamp.shape[0]
        nobs = len(self.__observer_names)
        pas = np.zeros((ntime, nobs))
        for t, pa in zip(unique_times, unique_pa):
            pas[utc_timestamp == t, :] = pa

        return pas

    def __apply_rotation(self, utc_timestamp, vis, a1, a2, clockwise=False):
        """
        Pads data and performs anticlockwise rotation by default
        """
        ## OMS: Ben had it in the opposite direction originally, but I'm sure this is right, and matches VLA results
        sign = -1 if clockwise else 1
        pa = sign * self.parallactic_angle(utc_timestamp)

        def mat_factory(pa, nchan, aindex, conjugate_transpose=False):
            def give_linear_mat(pa, nchan, aindex, conjugate_transpose=False):
                """ 2D rotation matrix according to Hales, 2017: 
                Calibration Errors in Interferometric Radio Polarimetry """
                c, s = np.cos(pa[:, aindex]).repeat(nchan), np.sin(pa[:, aindex]).repeat(nchan)
                N = pa.shape[0]
                if N == 0: 
                    return np.zeros((0, nchan, 2, 2)) # special case: no data for this baseline
                if conjugate_transpose:
                    return np.array([c, -s, s, c]).T.reshape(N, nchan, 2, 2)
                else:
                    return np.array([c, s, -s, c]).T.reshape(N, nchan, 2, 2)
                
            def give_circular_mat(pa, nchan, aindex, conjugate_transpose=False):
                """ phase rotation matrix according to Hales, 2017: 
                Calibration Errors in Interferometric Radio Polarimetry """
                e = np.exp(1.0j * pa[:, aindex]).repeat(nchan)
                ec = np.conj(e) # e * -1.0j
                null = np.zeros_like(e)
                N = pa.shape[0]
                if N == 0: 
                    return np.zeros((0, nchan, 2, 2)) # special case: no data for this baseline
                if conjugate_transpose:
                    return np.array([e, null, null, ec]).T.reshape(N, nchan, 2, 2)
                else:
                    return np.array([ec, null, null, e]).T.reshape(N, nchan, 2, 2)
            if self.feed_basis == "linear":
                return give_linear_mat(pa, nchan, aindex, conjugate_transpose)
            elif self.feed_basis == "circular":
                return give_circular_mat(pa, nchan, aindex, conjugate_transpose)
            else:
                raise RuntimeError("Factory does not understand %s feeds. This is a bug." % self.feed_basis)
        
        # pad visibility array if needed
        orig_vis_shape = vis.shape
        if vis.ndim == 3:
            pass
        elif vis.ndim == 2:
            vis = vis.reshape(vis.shape[0], vis.shape[1], 1)
        elif vis.ndim == 1:
            vis = vis.reshape(vis.shape[0], 1, 1)
        else:
            raise ValueError("VIS must be 3 dimensional array - nrow x nchan x ncorr")
        
        def pad(vis):
            if vis.shape[-1] == 4:
                padded_vis = vis.view()
            elif vis.shape[-1] == 2:
                padded_vis = np.zeros((vis.shape[0], vis.shape[1], 4), dtype=vis.dtype)
                padded_vis[:, :, 0] = vis[:, :, 0]
                padded_vis[:, :, 3] = vis[:, :, 1]
            elif vis.shape[-1] == 1:
                padded_vis = np.zeros((vis.shape[0], vis.shape[1], 4), dtype=vis.dtype)
                padded_vis[:, :, 0] = vis[:, :, 0]
                padded_vis[:, :, 1] = vis[:, :, 0]
                padded_vis[:, :, 2] = vis[:, :, 0]
                padded_vis[:, :, 3] = vis[:, :, 0]
            return padded_vis
        
        def unpad(vis, padded_vis):
            if vis.shape[-1] == 4:
                vis = padded_vis.view()
            elif vis.shape[-1] == 2:
                vis[:, :, 0] = padded_vis[:, :, 0]
                vis[:, :, 1] = padded_vis[:, :, 3]
            elif vis.shape[-1] == 1:
                vis[:, :, 0] = padded_vis[:, :, 0] # XX or RR only
            return vis
        
        nrow = 1000
        nchunk = int(np.ceil(vis.shape[0] / float(nrow)))
        for c in xrange(nchunk):
            lb, ub = c * nrow, min((c + 1) * nrow, vis.shape[0])
            if ub - lb == 0: 
                break
            padded_vis = pad(vis[lb:ub, :, :])
            p, q = np.triu_indices(np.max(list(set(a1[lb:ub]).union(set(a2[lb:ub]))))+1)  ## OMS: needs +1 or last baseline goes AWOL
            for bl in zip(p, q):
                blsel = np.logical_and(a1[lb:ub] == bl[0], a2[lb:ub] == bl[1])
                Pa1 = mat_factory(pa[lb:ub, :][blsel, :], 
                                  padded_vis.shape[1], 
                                  bl[0])
                Pa2H = mat_factory(pa[lb:ub, :][blsel, :], 
                                   padded_vis.shape[1], 
                                   bl[1], 
                                   conjugate_transpose=True)
                # Pp * V * Pq^H
                padded_vis[blsel, :, :] = np.matmul(Pa1, 
                                                    np.matmul(padded_vis[blsel, :, :].reshape((np.sum(blsel), 
                                                                                               padded_vis.shape[1],
                                                                                               2, 2)),
                                                              Pa2H)).reshape((np.sum(blsel),
                                                                              padded_vis.shape[1],
                                                                              4))
            vis[lb:ub, :, :] = unpad(vis[lb:ub, :, :].view(), padded_vis)
        vis.reshape(orig_vis_shape)
        return vis

    def rotate(self, utc_timestamp, vis, a1, a2, ack=True):
        """
        Rotates visibilties around the observer's third axis
        This can be applied to e.g MODEL_DATA because P is the first Jones in the chain
        and the fourier transform preserves rotation
        """
        if not self.__enable_rotation:
            return vis
        if ack:
            log.info("Applying P Jones to sky (precomputed)")
        return self.__apply_rotation(utc_timestamp, vis, a1, a2, clockwise=False)

    def derotate(self, utc_timestamp, vis, a1, a2, ack=True):
        """
        Rotates visibilties around the observer's third axis
        This can be applied to e.g CORRECTED_DATA because P
        is last to have a transpose 
        """
        if not self.__enable_derotation:
            return vis
        if ack:
            log.info("Applying P Jones to corrected data (precomputed)")
        return self.__apply_rotation(utc_timestamp, vis, a1, a2, clockwise=True)

    @property
    def enable_rotation(self):
        return self.__enable_rotation

    @enable_rotation.setter
    def enable_rotation(self, enable):
        if not enable:
            log.warn("Disabling parallactic rotation of model data as per user request")
        self.__enable_rotation = enable

    @property
    def enable_derotation(self):
        return self.__enable_derotation

    @enable_derotation.setter
    def enable_derotation(self, enable):
        if not enable:
            log.warn("Disabling parallactic derotation of corrected data as per user request")
        self.__enable_derotation = enable

    @property
    def feed_basis(self):
        return self.__feed_basis

    @feed_basis.setter
    def feed_basis(self, feed_basis):
        '''
        Sets feed basis
        Args:
        @feed_basis: Supports 'linear' or 'circular' bases
        '''
        log.info("Selecting '%s' feed bases" % feed_basis)
        if feed_basis not in ['linear', 'circular']:
            raise ValueError("Machine does not support {} feed basis. Only understands 'linear' and 'circular' bases")
        self.__feed_basis = feed_basis

    @property
    def field_centre(self):
        return self.__field_centre
    
    @field_centre.setter
    def field_centre(self, radec):
        '''
        Sets field centre RA DEC
        Args:
        @radec: Tuple of coordinates in degrees or ndarray of coordinates per observer in degrees
        '''
        if isinstance(radec, tuple) and len(radec) == 2:
            # Default mode is all observers look at the same direction
            # duplicate radec to all observers
            radec = np.tile(np.array([np.deg2rad(radec[0]), np.deg2rad(radec[1])]), 
                            (len(self.__observer_names), 1))
        # fly's eye otherwise -- position per observer
        elif isinstance(radec, np.ndarray) and \
             radec.ndim == 2 and \
             radec.shape[0] == len(self.__observer_names) and \
             radec.shape[1] == 2:
             radec = np.deg2rad(radec)
        else:
            raise ValueError("Expected coordinate tuple or nant x 2 ndarray for fly's eye observing")
        log.info("Changing field centre as follows:")
        for (aname, coord) in zip(self.__observer_names, radec):
            fmtcoord = [pq.quantity(fi, 'deg').formatted("ANGLE") for fi in np.rad2deg(coord)]
            log.info("\t%s\t%s\t%s\t%s" % (aname, self.__epoch, fmtcoord[0], fmtcoord[1]))
        self.__field_centre
        self.__field_centre = [pm.direction(self.__epoch,
                                            *(pq.quantity(fi, 'rad') for fi in fc))
                               for fc in radec]