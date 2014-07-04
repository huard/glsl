"""
Lecture des données de niveaux et débits d'EC.

David Huard, 2014

"""

"""Notes sur la base de données Hydat.

Téléchargement: 25 mars 2014.
Conversion sqlite: AccessDump.py
Commande SQL pour identifier les stations d'intérêt pour les débits:

SELECT STATION_NUMBER, STATION_NAME from "stations" WHERE STATION_NUMBER IN (
SELECT  STATION_NUMBER FROM "stations" WHERE "STATION_NAME" LIKE "%FLEUVE%"
INTERSECT
SELECT STATION_NUMBER FROM "stn_data_range" WHERE RECORD_LENGTH>30  AND DATA_TYPE='Q');
#======#
02MC010	SAINT-LAURENT (FLEUVE) - CENTRALE DE BEAUHARNOIS
02MC018	SAINT-LAURENT (FLEUVE) - CENTRALE DES CEDRES
02MC024	SAINT-LAURENT (FLEUVE)(CHENAL BEAUHARNOIS) LAC SAINT FRANCOIS
02MC029	SAINT-LAURENT (FLEUVE) AU BARRAGE DE L'ILE
02OA016	SAINT-LAURENT (FLEUVE) A LASALLE

02OJ032	SAINT-LAURENT (FLEUVE) A SOREL (INCLUANT RICHELIEU)
02OJ033	SAINT-LAURENT (FLEUVE) A SOREL (SANS RICHELIEU)
02OA039    SAINT-LOUIS (LAC) A POINTE-CLAIRE

International Great Lakes Datum (1985): #106, #431


"""





import numpy as np
import datetime as dt
import pandas as pd
import sqlite3
import h5py as h5
import os

HYDAT = '../data/Hydat_20140113.db'
ECSCEN = '../data/HYDRO_DATA_20140512.sqlite'
ECREG = 'lsl', 'lsp', 'mtl_lano'

stations_niveaux = {'Jetée #1 à Montréal':'02OA046',
                    'Varennes': '02OA050',
                    'Sorel':'02OJ022',
                    'Lac St-Pierre':'02OC016',
                    #'Trois-Rivières':'',
                    }

Bareas = dict(lacontario=8.4239E10, lacerie=8.6006E10, lachuron=1.91768E11, lacmichigan=1.73095E11, lacsuperior=2.10009E11, lacMHG=3.64863E11)



def NBS(freq='annual', stat=np.mean):
    """Return a dictionary holding the average NBS in the reference and future
    periods for each lake and simulation.
    """
    import ndict
    from scipy.io import matlab
    out = ndict.NestedDict()
    seasons = dict(winter=(12,1,2), spring=(3,4,5), summer=(6,7,8), fall=(9,10,11))
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # CRCM4
    M= matlab.loadmat('../data/NBS/CRCM4SerieNBS.mat', squeeze_me=True, struct_as_record=False)
    aliases = {}
    for per, key in zip(['ref', 'fut'], ['MSERIES_PR', 'MSERIES_FU']):
        D = M[key]
        aliases[per] = D._fieldnames

        for alias in D._fieldnames:
            d = getattr(D, alias)

            for lake in d._fieldnames:
                nbs = np.ma.masked_invalid( getattr(d, lake).nbs )
                if freq=='annual':
                    out[lake][alias] = stat(nbs[:,1:].mean(1))
                elif freq in seasons.keys():
                    out[lake][alias] = stat(nbs[:, seasons[freq]].mean(1))
                elif freq in months:
                    out[lake][alias] = stat(nbs[:, months.index(freq)+1])


    # OBS
    #Great Lakes basin: SUP (Lake Superior), MIC (Lake Michigan), HUR (Lake Huron), G(Georgian Bay), STC (Lake St. Clair), ERI (Lake Erie), and ONT (Lake Ontario).
    O = matlab.loadmat('../data/NBS/ObsGl.mat', squeeze_me=True, struct_as_record=False)['ObsGL']
    conv = dict(lacontario='ONT', lacerie='ERI', lachuron='HUR', lacmichigan='MIC', lacsuperior='SUP', lacMHG='MHG')
    for (lake, lid) in conv.items():
        nbs = getattr(O, lid).NBSPcpLake.data
        if freq=='annual':
            out[lake]['obs'] = stat(nbs[:,-1])/365.25
        elif freq in seasons.keys():
            out[lake]['obs'] = stat(nbs[:,seasons[freq]].mean(1))/(365.25/12)
        elif freq in months:
            out[lake]['obs'] = stat(nbs[:,months.index(freq)+1])/(365.25/12)

    return out# dict(zip(aliases['ref'], aliases['fut']))

def basins_weighted_NBS_average(nbs):
    """Return the basin weighted average."""
    import collections

    lakes = ['lacontario', 'lacMHG', 'lacerie', 'lacsuperior']
    A = sum([Bareas[l] for l in lakes])

    out = {}
    for alias in nbs.keylevel(1):
        out[alias] = sum([nbs[l][alias] * Bareas[l] for l in lakes])/A

    return out




def EC_H20(filename):
    """Return daily time series of water levels."""
    import codecs
    from matplotlib.dates import strpdate2num, num2date
    f = codecs.open(filename)

    convert = lambda x: num2date(strpdate2num('%Y/%m/%d')(x.decode()))
    ra = np.loadtxt(f, dtype={'names':('date', 'level'), 'formats':('datetime64[D]', 'f8')}, skiprows=8, delimiter=',', usecols=[0,1], converters={0:convert, 1:float})
    return pd.TimeSeries(ra['level'], ra['date'])

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


def _quartermonth_bounds(leap=False):
    """Returns the bounds for the quarter month day of the year.

    Notes
    -----
    Rules taken from Fan & Fay (2002).
    """
    if leap:
        mdays = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    limits = {28: [1, 8, 15, 22],
              30: [1,9,16,24],}
    limits[29] = limits[28]
    limits[31] = limits[30]

    qom = [0]
    for i in range(1,13):
        qom.extend((np.array(limits[mdays[i]]) + np.sum(mdays[:i])).tolist())

    return np.array(qom[1:], int)

def quartermonth_index(dti):
    """Return the year and quarter month index.

    Parameters
    ----------
    dti : DateTimeIndex
      Datetime values.
    """
    import calendar

    leap = [calendar.isleap(y) for y in dti.year]

    i = _quartermonth_bounds(False)
    li = _quartermonth_bounds(True)

    qom = np.digitize(dti.dayofyear, i)
    qom[leap] = np.digitize(dti.dayofyear[leap], li)

    return qom

def annual_min_dayly_ts(ts):
    """Return annual minimum of daily values.

    Parameters
    ----------
    ra: TimeSeries
      Daily time series.
    """

    # Group by year
    gr = ts.groupby(lambda x: x.year)

    # Count the number of values per year
    n = gr.count()

    # Find annual min
    ts_min = gr.min()

    # Set NaN where count is under threshold
    ts_min[n<300] = np.nan

    return ts_min


def qom2date(date, offset=0):
    """Convert a MultiIndex (year, qom) into date object."""

    qm = _quartermonth_bounds()

    out = []
    for i, d in enumerate(date):
        if type(d) == tuple:
            y, q = d
            out.append( dt.date(y+offset, 1, 1) + dt.timedelta(days=int(qm[q-1])-1) )
        else:
            if hasattr(date, 'index'):
                out.append( dt.date(date.index[i]+offset, 6, 1))
            else:
                out.append(np.nan)

    return out

def group_qom(ts, dt=False):
  """Take TimeSeries and return a grouped object."""
  return ts.groupby([lambda x: x.year, quartermonth_index])

def qom2ts(a):
    """Convert a QOM series into a TimeSeries."""
    i = pd.DatetimeIndex( qom2date(a.index.to_series().values) )
    return pd.TimeSeries(data=a.values, index=i)


def annual_min_qom_ts(ts):
    """Return annual minimum of quarter-monthly values.

    Parameters
    ----------
    ra: TimeSeries
      Daily time series.

    TODO: count values in QOM to assert minimum of 4 days.
    """

    # Group by year and QoM
    gr = ts.groupby([lambda x: x.year, quartermonth_index])

    # Average over QoM
    ts_mean = gr.mean()

    # Group by year
    gro = ts_mean.groupby(level=0)

    # Count the number of values per year
    n = gro.count()

    # Find annual min
    ts_min = gro.min()
    # Set NaN where count is under threshold
    ts_min[n<40] = np.nan

    return ts_min

def get_station_meta(sid):
    """Get the meta data for the station."""
    fields = get_fields('stations')
    values = query('stations', 'STATION_NUMBER', sid)
    for row in values:
        return dict(zip(fields, row))

def query(table, field, value, path=HYDAT):
    """General purpose query."""
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        CMD = "SELECT * FROM {0} WHERE {1} LIKE ?".format(table, field)
        rows = cur.execute(CMD, (value,))
        return list(rows)


def get_fields(table, path=HYDAT):
    """Return the column names."""
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()

        rows = cur.execute("PRAGMA table_info({0})".format(table))
        fields = [row[1] for row in rows]
        return fields


def get_hydat(sid, var='Q'):
    """Return the daily flows or levels at the station.

    Parameters
    ----------
    sid : str
      Station id
    var : Q or H
      Flows or levels.
    """

    with sqlite3.connect(HYDAT) as conn:
        cur = conn.cursor()

        series = []
        if var == 'Q':
            table = "dly_flows"
            i = 11
        elif var == 'H':
            table = "dly_levels"
            i = 12
        else:
            raise ValueError("Variable not recognized")

        fields = get_fields(table)

        CMD = "SELECT * FROM {0} WHERE STATION_NUMBER = ?".format(table)

        for row in cur.execute(CMD, (sid,)):
            meta = dict(zip(fields[:7], row[:7]))

            x = row[i::2]
            dates = pd.date_range('{0}-{1}-01'.format(meta['YEAR'], meta['MONTH']), periods=meta['NO_DAYS'], freq='D')
            series.append( pd.TimeSeries(x[:meta['NO_DAYS']], dates))

        S = pd.concat(series).sort_index()#.resample('D')
        return S.convert_objects(convert_numeric=True)


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
        Y.append( x[i,0].astype(int).repeat(48) )
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
