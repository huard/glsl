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




"""





import numpy as np
import datetime as dt
import pandas as pd
import sqlite3

def EC_H20(filename):
    import codecs
    from matplotlib.dates import strpdate2num, num2date
    f = codecs.open(filename)
    
    convert = lambda x: num2date(strpdate2num('%Y/%m/%d')(x.decode()))
    return np.loadtxt(f, dtype={'names':('date', 'level'), 'formats':('datetime64[D]', 'f8')}, skiprows=8, delimiter=',', usecols=[0,1], converters={0:convert, 1:float})


def quartermonth_bounds(leap=False):
    if leap:
        mdays = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    else:
        mdays = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    
    qom = []
    for i in range(12):
        qom.extend(np.around(np.linspace(mdays[i], mdays[i+1], 4, False)).tolist())
        
    return np.array(qom, int)
     
def quartermonth_index(dti):
    """Return the year and quarter month index.
    
    Parameters
    ----------
    dti : DateTimeIndex
      Datetime values.
    """
    import calendar
        
    leap = [calendar.isleap(y) for y in dti.year]
    
    i = quartermonth_bounds(False)
    li = quartermonth_bounds(True)
    
    qom = np.digitize(dti.dayofyear, i)
    qom[leap] = np.digitize(dti.dayofyear[leap], li)
    
    return qom

def annual_min_dayly_ts(ra):
    """Return annual minimum of daily values.
    
    Parameters
    ----------
    ra: record array
      Time series. 
    """
    
    ts = pd.Series(ra['level'], ra['date'])
    
    # Group by year
    gr = ts.groupby(lambda x: x.year)
    
    # Count the number of values per year
    n = gr.count()
    
    # Find annual min
    ts_min = gr.min()
    
    # Set NaN where count is under threshold
    ts_min[n<300] = np.nan
    
    return ts_min

def annual_min_qom_ts(ra):
    """Return annual minimum of quarter-monthly values.
    
    Parameters
    ----------
    ra: record array
      Time series. 
    """
    ts = pd.Series(ra['level'], ra['date'])
    
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
    
    with sqlite3.connect('../data/Hydat_20140113.db') as conn:
        cur = conn.cursor()
            
        CMD = "SELECT * FROM stations WHERE STATION_NUMBER = ?"
        rows = cur.execute(CMD, (sid,))
        for row in rows:
            return dict(zip(fields, row))

def query(table, field, value):
    """General purpose query."""
    with sqlite3.connect('../data/Hydat_20140113.db') as conn:
        cur = conn.cursor()
        CMD = "SELECT * FROM {0} WHERE {1} = ?".format(table, field)
        rows = cur.execute(CMD, (value,))
        return list(rows)
        

def get_fields(table):
    """Return the column names."""
    with sqlite3.connect('../data/Hydat_20140113.db') as conn:
        cur = conn.cursor()
            
        rows = cur.execute("PRAGMA table_info({0})".format(table))
        fields = [row[1] for row in rows]
        return fields
            
            
def get_dly(sid, var='Q'):
    """Return the daily flows or levels at the station.
    
    Parameters
    ----------
    sid : str
      Station id
    var : Q or H
      Flows or levels.
    """
    
    with sqlite3.connect('../data/Hydat_20140113.db') as conn:
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
            series.append( pd.TimeSeries(x[:meta['NO_DAYS']], dates) ) 
                            
        return pd.concat(series).sort_index()
            
    
    
