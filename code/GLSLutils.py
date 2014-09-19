import numpy as np
import datetime as dt
import pandas as pd


def ordinal_qom(ts):
    """Given a QoM MultiIndex, return an array of ordinals corresponding to the date."""
    import operator
    toord = operator.methodcaller('toordinal')
    d = qom2date(ts.index.to_series().values)
    return np.array(list(map(toord, d)))


def ordinal_ts(ts):
    """Return the ordinal value of a time series date index."""
    import operator
    toord = operator.methodcaller('toordinal')
    return np.array(list(map(toord, ts.index)))

def fitfunction(q, l):
    """Given streamflows and levels, return a function of the form ax**b that
    fits the q-l relation.
    """
    from scipy import optimize
    c = optimize.fmin(lambda C: np.sum((C[0] * q ** C[1] - l)**2), [1,1])
    return lambda x: c[0]*x**c[1]



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
