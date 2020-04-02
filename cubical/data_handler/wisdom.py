from cubical.tools import logger, ModColor
log = logger.getLogger("wisdom")


def estimate_mem(data_handler, tile_list):

    wis_nrow = tile_list[0].last_row0 - tile_list[0].first_row0
    wis_nchan = data_handler.all_freqs.size
    wis_nant = data_handler.nants
    wis_ncorr = tile_list[0].ncorr

    print(wis_nant, wis_nchan, wis_ncorr, wis_nrow, file=log)

    return