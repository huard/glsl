"""
All IO relation to Fan & Fay
"""
import numpy as np
import pandas as pd
import os, re
import GLSLutils


offset = {'bc': 1961,
          'ww': 2039,
          'wd': 2039}


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
        m = re.match('UNITS:\s*(\w*)(m3s|m3/s|n|m)', txt)
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

def FF_flow(site, scen='bc', yo=0, up=False):
    """Return the flow at the given site provided by Fan & Fay.

    Parameters
    ----------
    site : {rich, fran, mau, dpmi, stl, ont}
      River or site name.
    scen : {bc, ww, wd}
      Climate scenarios: base case, warm & wet, warm & dry.
    yo : int
      Year offset.
    up : bool
      Use updated values from the Upper Great Lakes study.

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


    if up:
        if scen != 'bc':
            raise ValueError('Only base case scenario available for updates.')

        DIR = '../data/FF/Update_58DD/Plan58DD_Levels_Flow'
        if site == 'ont':
            fn = os.path.join(DIR, 'Qont_1900_2008.58DD')
        elif site == 'stl':
            fn = os.path.join(DIR, 'Stlqmq.58DD')
        elif site == 'lsp':
            fn = os.path.join(DIR, 'stprQL.58DD')
        elif site == 'var':
            fn = os.path.join(DIR, 'vrnQL.58DD')
        elif site in tributaries:
            DIR = '../data/FF/Update_58DD/Plan58DD_Tributaries'
            if site == 'fran':
                ns = 'stfr'
            elif site == 'mau':
                ns = 'stmrc'
            elif site in ['dpmi', 'rich']:
                ns = site

            fn = os.path.join(DIR, '{0}qmq.dbf'.format(ns))

    else:
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


def FF_K(site, scen='bc', y0=0):
    """Return the roughness factor K."""
    if scen not in ['bc', 'wd', 'ww']:
        raise ValueError

    DIR = '../data/FF/Input_CC'
    fn = os.path.join(DIR, '{0}.{1}'.format(site, scen))
    if site == 'lsp':
        fn = os.path.join(DIR, 'stprnqm.nov')
        return _loadFF(fn, -1899+y0)

    return _loadFF(fn, y0)


def FF_tidal(scen='bc', y0=0):
    DIR = '../data/FF/Input_CC'
    fn = os.path.join(DIR, 'tidal.{0}'.format(scen))
    return _loadFF(fn, y0)

def FF_level(site, scen='bc', y0=0, up=False):
    """Return the water level at the given site provided by Fan & Fay.

    Paramters
    ---------
    site : {mtl, ont, pcl, srl, lsp}
      Site names: Montreal Jetty #1, Lake Ontario, Pointe Claire, Sorel,
      Lac St-Pierre.
    scen : {bc, ww, wd}
      Climate scenarios: base case, warm & wet, warm & dry.
    yo : int
      Year offset.
    up : bool
      Use updated values from the Upper Great Lakes study.
    """
    if up:
        if scen != 'bc':
            raise ValueError ("Only base case scenario is available.")
        scen = '58DD'

        DIR = '../data/FF/Update_58DD/Plan58DD_Levels_Flow'
        if site == 'mtl':
            name = 'JTY1ql'
        elif site == 'srl':
            name = 'srlQL'
        elif site == 'stl':
            name = 'stlqml'
        elif site == 'ont':
            name = 'Level_1900_2008'
        elif site == '3rv':
            name = 'rivQL'
        elif site == 'lsp':
            name = 'stprQL'
        elif site == 'var':
            name = 'vrnQL'

    else:
        DIR = '../data/FF/Plan58DD_CC'
        if site == 'ont':
            name = 'Level_Feb06'
        elif site == 'mtl':
            name = 'mtl_level_Plan58DD_CC'
        elif site == 'pcl':
            name = 'pcl_level_Plan58DD_CC'

    fn = os.path.join(DIR, '{0}.{1}'.format(name, scen))
    return _loadFF(fn, y0)

def stage_func(site,):
    """Return the equation from Fan & Fay (2002) relating stage to
    discharge at locations on the St. Lawrence.

    Parameters
    ----------
    site : {'mtl', 'var', 'srl', 'lsp', trois'}
      Name of control point.

    Returns
    -------
    func : function
      Function computing level from flow series

    Notes
    -----

    Control points:
     * Jetée #1 (15520) (02OA046)
     * Varennes (155660) (02OA050) 45.684º N, 73.443º W
     * Sorel (15930) (02OJ022)  46.047º N, 73.115º W
     * Lac St-Pierre (15975) (02OC016) 46.194º N, 72.895º W
     * Trois-Rivières (03360) (?) 46.3405º N, 72.539167º W

    The function will then accept a vector of flows from
     * Lac St-Louis (02OA016, 02OA024)
     * Des Prairies & Milles-Iles (02OA004, 02OA003)
     * Richelieu (02OJ007)
     * St-François (02OF019)
     * St-Maurice (02NG005)

    """
    regcoefs = {'mtl':[(.001757, .000684, 0, 0.001161, 0.000483), 0.6587, 0.9392],
                'var':[(0.001438, 0.001377, 0, 0.001442, 0.000698), 0.6373, 1.0578],
                'srl':[(0.001075, 0.001126, 0, 0.001854, 0.000882), 0.6331, 1.277],
                'lsp':[(0.000807, 0.001199, 0, 0.001954, 0.000976), 0.6259, 1.4722],
                'trois':[(.000584, .00069, .000957, .001197, .000787), .7042, 1.5895],
                #'tr':[(.000589, .000727, .00102, .001158, .000815), 0.6981, 1.5919],
                }

    c, h, t = regcoefs[site.lower()]

    def func(Q, tidal):
        """Return level computed from tributaries' flow and tidal component.
        Note that the roughness factor accounting for ice effects is
        expected to be included in the flow Q.
        """
        a = np.dot(c, Q)**h
        t1 = pd.Series(data=a, index=Q.axes[1])
        return t1 + t * tidal
    return func

def level_series_QH(site, scen='bc'):
    """Return the levels from the Fan & Fay scenarios.

    Parameters
    ----------
    site : {'mtl', 'var', 'srl', 'lsl', 'lsp', 'trois'}
      Name of control point.

    scen : {'bc', 'wd', 'ww'}
      Scenario name: Base Case, Warm & Dry, Warm & Wet.

    """
    qs = 'stl', 'dpmi', 'rich', 'fran', 'mau'
    y0 = offset[scen]
    Q = pd.DataFrame([FF_flow(q, scen, y0) for q in qs])

    K = FF_K(site, scen, y0)
    T = FF_tidal(scen, y0)

    # Add one year to the K and T records (Q also misses one year... shit)
    #K = pd.concat([K, GLSLutils.select_and_shift(K, slice(y0+1,y0+2), 29)])
    #T = pd.concat([T, GLSLutils.select_and_shift(T, slice(y0+1,y0+2), 29)])

    f = stage_func(site)
    ts = f(K*Q, T)

    return pd.Series(ts, K.index)

def get_flow_sorel(scen='bc'):
    """Sum the flow from tributaries to get the flow at Sorel from the
    F&F flows.

    Note
    ----
    By neglecting the flows from smaller tributaries, isn't there is a risk of
    underestimating the levels, because we are systematically underestimating
    the flows?

    OBSOLETE, see total_flow
    """
    qs = 'stl', 'dpmi'
    yo = offset[scen]
    Q = [FF_flow(q, scen, yo)for q in qs]
    return sum(Q)

def total_flow(site, scen='bc'):
    qs = {'srl':['stl', 'dpmi'],
          'lsp':['stl', 'dpmi', 'rich', 'fran'],
          'ont':['ont']}

    yo = offset[scen]
    Q = [FF_flow(q, scen, yo) for q in qs[site]]
    return sum(Q)

def PCL(scen='bc'):
    """Updated Relationship between the Outflow from Lac St. Louis and the level
    at the Pointe Claire gauge.

    The relationship developed between the quarter-monthly water levels recorded
    at the Pointe Claire gauge site and the Lac Saint-Louis outflows is as follows.

    HPCL = 16.57 + ( FSTL *QSTL /604.) ^0.580
    Where: HPCL: the water level of Lac St Louis at Pointe Claire, m IGLD 1985
    FSTL = the ice factor for Lac St Louis, dimensionless
    QSTL = outflow from Lac St Louis, m3/s

    Reference
    ---------
    Fan, Yin and Fay, David. January 2003 Estimation of Ice effects on Lac
    Saint-Louis Outflows and Levels. Unpublished report. Great Lakes -
    St. Lawrence Regulation Office, Meteorological Service of Canada - Ontario
    Region, Environment Canada, Cornwall ON.
    """
    q = FF_flow('stl', scen, offset[scen])
    f = FF_K('pcl', scen, offset[scen])
    return 16.57 + (f*q/604.)**.58
