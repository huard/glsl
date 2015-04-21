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

def nodays(leap=False):
    """Return the number of days in each QoM."""
    if leap:
        mdays = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    n = {28: [7,7,7,7],
         29: [7,7,7,8],
         30: [8, 7, 8, 7],
         31: [8, 7, 8, 8]}

    ndays = []
    for m in mdays:
        ndays.extend(n[m])
    return ndays

def qomdays(dti):
    """Return the number of days per QoM for the index.

    Parameters
    ----------
    dti : DateTimeIndex
      Datetime values.
    """
    import calendar

    nd = {True: nodays(True), False: nodays(False)}

    out = []
    for yv,qi in zip(dti.get_level_values(0), dti.labels[1]):
        out.append(nd[calendar.isleap(yv)][qi])

    return np.array(out)


def _quartermonth_bounds(leap=False):
    """Returns the bounds for the quarter month day of the year.

    Notes
    -----
    Rules taken from Fan & Fay (2002).
    """
    d = [1,] + nodays(leap)
    return  np.cumsum(d)[:-1]

def quartermonth_index(dti):
    """Return the year and quarter month index.

    Parameters
    ----------
    dti : DateTimeIndex
      Datetime values.
    """
    import calendar

    leap = np.array([calendar.isleap(y) for y in dti.year])

    i = _quartermonth_bounds(False)
    li = _quartermonth_bounds(True)

    qom = np.empty(len(dti))
    if leap.any():
        qom[leap] = np.digitize(dti.dayofyear[leap], li)
    if (~leap).any():
        qom[~leap] = np.digitize(dti.dayofyear[~leap], i)

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

def select_and_shift(ts, before, after, offset=0):
    b = ts.reindex(ts.index.truncate(before=before, after=after))
    b.index = pd.MultiIndex.from_arrays([b.index.get_level_values(0).values+offset, b.index.get_level_values(1)], names=b.index.names)
    return b

def qom2date(date, offset=0):
    """Convert a MultiIndex (year, qom) into date object."""
    import calendar

    qm = {}
    qm[True] = _quartermonth_bounds(leap=True)
    qm[False] = _quartermonth_bounds(leap=False)

    out = []
    for i, d in enumerate(date):
        if type(d) == tuple:
            y, q = d
            leap = calendar.isleap(y)
            out.append( dt.date(y+offset, 1, 1) + dt.timedelta(days=int(qm[leap][q-1])-1) )
        else:
            raise "hein?"
            if hasattr(date, 'index'):
                out.append( dt.date(date.index[i]+offset, 6, 1))
            else:
                out.append(np.nan)

    return out

def qom2season(date):
    """Return the season of the QoM."""

    q = date[1]
    if q in range(9,21):
        return 'Spring'
    elif q in range(21, 33):
        return 'Summer'
    elif q in range(33, 45):
        return "Fall"
    elif q in list(range(45, 49)) + list(range(1,9)):
        return 'Winter'
    else:
        raise ValueError("QoM not recognized.")


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
