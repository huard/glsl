"""
All IO related to the HYDAT dataset
"""
HYDAT = '../data/Hydat_20140113.db'

import sqlite3
import numpy as np
import pandas as pd


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
