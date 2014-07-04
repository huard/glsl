"""
All IO relation to Fan & Fay
"""
import numpy as np
import pandas as pd
import os, re

def _loadFF(fn, yo=0):
    """
    Read the Fortran formatted file from Fan & Fay.

    Parameters
    ----------
    fn : str
      File name.
    yo : int
      Year offset.
    """
    import re
    with open(fn) as f:
        for i in range(4):
            txt = f.readline()
        m = re.match('UNITS:(\w*)(m3s|n|m)', txt)
        if m:
            if m.groups()[0] == '':
                scale = 1
            else:
                scale = int(m.groups()[0])
            units = m.groups()[1]
        else:
            raise ValueError('Could not parse units.')


    x = np.loadtxt(fn, skiprows=6)
    n = x.shape[0]
    V = []; Y = []; W = []
    for i in range(0, n, 4):
        V.append( x[i:i+4, 2:].T.ravel() )
        Y.append( (x[i,0].astype(int)+yo).repeat(48) )
        W.append( np.arange(1,49) )

    V = np.concatenate(V)
    Y = np.concatenate(Y)
    W = np.concatenate(W)

    mi = pd.MultiIndex.from_arrays([Y,W], names=('Year', 'QTM'))
    ts = pd.Series(V, mi)*scale

    with open(fn) as f:
        line = f.readline()[1:]
        i = line.lower().find('quarter')

        ts.name = line[:i-1]

    ts.units = units
    return ts

def FF_flow(site, scen='bc', yo=0):
    """Return the flow at the given site provided by Fan & Fay.

    Parameters
    ----------
    site : {lsl, rich, fran, mau, dpmi, stl, ont}
      River or site name.
    scen : {bc, ww, wd}
      Climate scenarios: base case, warm & wet, warm & dry.
    yo : int
      Year offset.

    Notes
    -----
    * stl: Lac St-Louis
    * rich: Richelieu
    * fran: St-Francois
    * mau: St-Maurice
    * dpmi: Des Prairies Milles Iles
    * ont: Lake Ontario
    """

    tributaries = 'mau', 'fran', 'rich', 'chat', 'bat'

    if site in tributaries:
        DIR = '../data/FF/Input_CC'
        addon = ''
    else:
        DIR = '../data/FF/Plan58DD_CC'
        if site in ['dpmi', 'stl']:
            addon = '_Plan58DD_CC'
        elif site == 'ont':
            addon = '_Feb06'

        if site in ['stl', 'ont']:
            site = 'Q' + site

        elif site == 'dpmi':
            site = site+'qmq'


    fn = os.path.join(DIR, '{0}{1}.{2}'.format(site, addon, scen))
    return _loadFF(fn, yo)

def FF_K(site, scen='bc'):
    """Return the roughness factor K."""
    DIR = '../data/FF/Input_CC'
    fn = os.path.join(DIR, '{0}.{1}'.format(site, scen))
    return _loadFF(fn)


def FF_tidal(scen='bc'):
    DIR = '../data/FF/Input_CC'
    fn = os.path.join(DIR, 'tidal.{0}'.format(scen))
    return _loadFF(fn)

def FF_level(site, scen='bc'):
    """Return the water level at the given site provided by Fan & Fay.

    Paramters
    ---------
    site : {mtl, ont, pcl, srl, lsp}
      Site names: Montreal Jetty #1, Lake Ontario, Pointe Claire, Sorel,
      Lac St-Pierre.

    scen : {bc, ww, wd}
      Climate scenarios: base case, warm & wet, warm & dry.
    """
    DIR = '../data/FF/Plan58DD_CC'
    if site == 'ont':
        name = 'Level_Feb06'
    elif site == 'mtl':
        name = 'mtl_level_Plan58DD_CC'
    elif site == 'pcl':
        name = 'pcl_level_Plan58DD_CC'

    fn = os.path.join(DIR, '{0}.{1}'.format(name, scen))
    return _loadFF(fn)
