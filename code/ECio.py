"""
All IO related to EC scenarios.
"""
import sqlite3
import numpy as np
import h5py as h5
import pandas as pd
import os

ECSCEN = '../data/HYDRO_DATA_20140512.sqlite'
ECREG = 'lsl', 'lsp', 'mtl_lano'

def Q_Sorel(freq='qtm'):
    """
    Débit reconstitué à Sorel par EC selon la méthode Bouchard et Morin
    (2000).

    Parameters
    ----------
    freq : {qtm, week}
      Frequency, either quarter of month or weekly.

    Returns
    -------
    out : Series
      Series of flow at Sorel.

    Notes
    -----
    The weeks seem to be starting on Jan 1st, not on a Sunday.
    """
    import xlrd

    with xlrd.open_workbook('../data/Q-Sorel.xlsx') as wb:

        if freq == 'qtm':
            ws = wb.sheet_by_name('Q-Sorel-AvgQtMois')
        elif freq=='week':
            ws = wb.sheet_by_name('Q-Sorel-AvgHebdo')
        else:
            raise ValueError("Option {0} not recognized.".format(freq))

        Y = np.array(ws.col_values(0)[1:], int)
        W = np.array(ws.col_values(1)[1:], int)
        V = ws.col_values(2)[1:]

        # Check that the record is continuous
        #assert set(np.diff(Y)) == set([0,1])
        #assert np.all(np.nonzero(np.diff(W) < 0)[0] == np.nonzero(np.diff(Y)==1)[0])

        mi = pd.MultiIndex.from_arrays([Y,W], names=('Year', 'QTM' if freq=='qtm' else 'Week'))
        return pd.Series(V, mi)


def get_scen(no, reg='mtl_lano', variables=None):
    """Return the coordinates and values for a given scenario.

    Parameters
    ----------
    no : int
      Scenario index (1-8)
    reg : {mtl_lano, lsl, lsp}
      Region code: Mtl-Lanaudiere, Lac St-Louis, Lac St-Pierre. None will
      return points for all three regions.

    Return
    ------
    x, y, z, depth, velocity
    """

    if reg is None:
        reg = ECREG
    else:
        assert reg in ECREG
        reg = [reg]

    if variables is None:
        variables = 'X, Y, Z, PROFONDEUR, MOD_VITESSE'
    else:
        variables = ', '.join(variables)

    with sqlite3.connect(ECSCEN) as conn:
        cur = conn.cursor()
        out = []
        n = len(out)
        for r in reg:
            if reg == 'mtl_lano':
                fmt = "'{0}P'"
            else:
                fmt = "{0}P"
            CMD = "SELECT {0} FROM data_{1} WHERE SCENARIO = '{2}P'".format(variables, r, no)
            rows = cur.execute(CMD)#, (fmt.format(no),))
            out.extend(rows)

            if len(out) - n  == 0:
                raise ValueError("No data in query: {0}".format(CMD))


    return np.array(out, float).T

def convert_scen_to_h5():
    """Convert the sqlite database to HDF5 for speed."""

    with h5.File('../data/HYDRO_DATA_EC.h5', mode='w') as F:

        for reg in ECREG:
            G = F.create_group(reg)

            x, y, z = get_scen(1, reg, ('X', 'Y', 'Z'))
            G.create_dataset('xyz', data=np.array([x, y, z]))
            D = G.create_dataset('depth', shape=(8,len(x)))
            S = G.create_dataset('speed', shape=(8,len(x)))

            for scen in range(8):
                d, s = get_scen(scen+1, reg, variables=('PROFONDEUR', 'MOD_VITESSE'))
                D[scen,:] = d
                S[scen,:] = s

def EC_pts(reg=None):
    """Return coordinates of grid points (x,y,z).

    Parameters
    ----------
    reg : str, list
      Name of region or list of regions to obtain the coordinates for. If None,
      return a dict with all regions.

    Returns
    -------
    out : x,y,z if reg is the region's name, otherwise a dict of (x,y,z) keyed
    by region name.

    """
    out = {}
    if reg is None:
        regions = ECREG
    elif type(reg) == str:
        regions = [reg,]

    with h5.File('../data/HYDRO_DATA_EC.h5') as F:
        for r in regions:
            G = F.get(r)
            out[r] = G.get('xyz')[:]

    if type(reg) == str:
        return out[reg]
    else:
        return out

def EC_depth(reg=None, scen=None):
    """Return water depth."""
    out = {}
    if reg is None:
        regions = ECREG
    elif type(reg) == str:
        regions = [reg,]

    s = scen-1 if scen else slice(scen)

    with h5.File('../data/HYDRO_DATA_EC.h5') as F:
        for r in regions:
            G = F.get(r)
            out[r] = G.get('depth')[s]

    if type(reg) == str:
        return out[reg]
    else:
        return out



def MTM8():
    """Convert Modified Transverse Mercatorimport Zone 8 coordinates to lat-lon.

    # NAD83 / MTM zone 8 Québec
    <42104> +proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.999900 +x_0=304800 +y_0=0 +ellps=GRS80 +units=m +no_defs  no_defs <>
    """
    from mpl_toolkits.basemap import pyproj
    return pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=-73.5 +k=0.999900 +x_0=304800 +y_0=0 +ellps=GRS80 +units=m +no_defs")

la_prairie = -73.500732, 45.417809
ile_perrot =  -73.947432, 45.399887

def level(lon, lat, scen, reg=None):
    from scipy.interpolate import LinearNDInterpolator

    # Find native coordinates
    x, y = MTM8()(lon, lat)

    # Get scenario data
    X, Y, Z, D, V = get_scen(scen, reg)

    # Create triangulation
    I = LinearNDInterpolator(list(zip(X, Y)), list(zip(Z, D)))
    return I((x,y))
