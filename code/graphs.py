import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import basemap
from matplotlib.ticker import Formatter
import pandas as pd
import datetime as dt

import GLSLio
import analysis
from imp import reload
reload(GLSLio)
reload(analysis)

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Droid Sans'
c = '#00425C'
CC = dict(bc='Historique', ww='Chaud et humide', wd='Chaud et sec')


# see tricontour_smooth_delaunay.py

# Attention: les trois domaines n'ont pas l'air d'être sur le même datum (ça parait à la jonction du Lac St-Pierre)

# Haxby color palette
ncolors=11;
_haxby=np.array([[ 37, 57, 175], [40, 127, 251], [50, 190, 255], [106, 235, 255], [138, 236, 174], [205, 255, 162], [240, 236, 121], [255, 189, 87], \
    [255, 161, 68], [255, 186, 133], [255, 255, 255]])/255.;
haxby = mpl.colors.LinearSegmentedColormap.from_list('haxby', _haxby[::-1])

class QOMFormatter(Formatter):
  def __init__(self, seq):
    self.seq = seq
  def __call__(self, x, pos=None):
    if pos is None or pos >= len(self.seq):
      return ''
    else:
      try:
        print (x, self.seq.index[x])
        return str(self.seq.index[x][0])
      except:
        return ''

def strip(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_position(('outward',10))
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

def plot_FF_flow(site):
    """Plot time series of the flow.
    """
    from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
    # Plot only the first 29 years
    # The next years are not a perfect copy of the first years, I suspect because
    # of the storage effect.
    #if scen == 'bc':
    yo = 1961
    c = dict(bc='#393737', wd='#ab4918', ww='#106b86')

    fig, ax = plt.subplots(1,1, figsize=(8,4), facecolor='w')

    fig.subplots_adjust(right=.97, top=.95)
    strip(ax)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.autoscale(enable=True, tight=True)
    ax.set_ymargin(.05)

    # QOM series
    for scen in ['bc', 'ww', 'wd']:
        ts = GLSLio.FF_flow(site, scen, yo)[:29*48]
        ts_d = [d.toordinal() for d in GLSLio.qom2date(ts.index.to_series().values)]

        # Annual min
        g = ts.groupby(level=0)
        tm = g.min()
        itm = g.idxmin()
        tm_d = [d.toordinal() for d in GLSLio.qom2date(itm)]


        if scen == 'bc':
            ax.plot_date(ts_d, ts.values, '-', color='#393737', alpha=.5)

        ax.plot_date(tm_d, tm.values, '-', color=c[scen], marker='o',  mfc='w', mec=c[scen], label=CC[scen], lw=1.5, mew=1.5)

    leg = ax.legend(loc='upper left', frameon=False, numpoints=1, title='Minimum annuel', fontsize='small')
    ax.set_ylabel("Débit [m³/s]")


    return fig, ax



def plot_station_qom_levels(sid):
  """Graphic of the mean QOM level time series."""
  ts = GLSLio.get_hydat(sid, 'H')

  # Average at the QOM scale
  gr = GLSLio.group_qom(ts)
  ts = gr.mean()


  fig, ax = plt.subplots(1, 1, figsize=(10, 4))
  fig.subplots_adjust(bottom=.1, right=.95, left=.1)
  ts.plot(ax=ax, color=c, lw=1.5)
  ts.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)

  ax.grid(ls='-', lw=.1, color='#777777')
  ax.grid(axis='x')

  ax.set_ylabel("Niveau [m]")


  ax.set_xlim(0, len(ts))

  years = ts.index.levels[0]
  if len(years) > 20:
    xt = []; t = []
    y1, y2 = np.around(years[0]+4,-1), np.around(years[-1]-4,-1)
    for i, y in enumerate(np.arange(y1, y2+1, 10)):
       xt.append(ts.index.get_loc(y).start)
       t.append(str(y))

  ax.set_xticks(xt)
  ax.set_xticklabels(t)

  return fig, ax

def ordinal(ts):
    """Return the ordinal value of a time series date index."""
    import operator
    toord = operator.methodcaller('toordinal')
    return list(map(toord, ts.index))

def plot_station_qom_level_stats(sid):
    """Graphic of the statistics of the level time series."""
    import itertools

    dts = GLSLio.get_hydat(sid, 'H').resample('D')

    # Average at the QOM scale
    gr = GLSLio.group_qom(dts)
    n = gr.count()
    qs = pd.Series(data=np.where(n >= 4, gr.mean(), np.nan), index=n.index)

    g = qs.groupby(level=0)
    m = [g.count() < 30]
    set_mask = lambda x: np.where(g.count() > 30, x, np.nan)

    ts = GLSLio.qom2ts(qs)

    years = g.mean().index
    i = [dt.datetime(y, 6, 1) for y in years]
    fb = zip(years, itertools.cycle([1,]))

    tsm = pd.TimeSeries(data=set_mask(g.mean().values), index=i)
    tsmin = pd.TimeSeries(data=set_mask(g.min().values), index=GLSLio.qom2date(g.idxmin()))
    tsmax = pd.TimeSeries(data=set_mask(g.max().values), index=GLSLio.qom2date(g.idxmax()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='w')
    fig.subplots_adjust(bottom=.1, right=.97, left=.07, top=.9)
    strip(ax)


    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.autoscale(enable=True, tight=True)
    ax.set_ymargin(.05)

    ax.plot_date(ordinal(ts), ts.values, '-', lw=.5, color='#393737', alpha=.5)
    ax.plot_date(ordinal(tsm), tsm.values, '-', marker='o', color='#393737', mfc='w', mec='#393737', mew=1.5, lw=1.5, label='Moyenne')
    ax.plot_date(ordinal(tsmin), tsmin.values, '-', marker='o', color='#810202', mfc='w', mec='#810202', mew=1.5, lw=1.5, label='Minimum')
    ax.plot_date(ordinal(tsmax), tsmax.values, '-', marker='o', color='#3866b7', mfc='w', mec='#3866b7', mew=1.5, lw=1.5, label='Maximum')


    #ax.fill_between(tsm.index, tsmax, tsmin, color="#333333", alpha=.1)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.set_ylabel("Niveau [m]")
    ax.legend(loc='lower right', ncol=3, frameon=False, numpoints=1, bbox_to_anchor=(1,.95), fontsize='small', title='Statistiques annuelles')
    return fig, ax



def plot_Mtl_annual_minimum_qom_levels():
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15520-01-JAN-1900_slev.csv')

    ts_min = GLSLio.annual_min_qom_ts(ra)

    # Graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.subplots_adjust(bottom=.1, right=.95, left=.1)
    ts_min.plot(ax=ax, color=c, lw=1.5)
    ts_min.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ylim = ax.get_ylim()
    dy = ylim[1] - ylim[0]

    ax.set_ylabel("Niveau [m]")
    ax.set_title("Minimum annuel du niveau d'eau à la jetée #1 du port de Montréal")

    for y in [1934, 1965, 2001, 2007, 2012]:
        ax.annotate(str(y), (y, ts_min[y]+dy*.02), (y, ts_min[y]+dy*.8), ha='center', size='small', arrowprops=dict(arrowstyle='-', fc='w', ec='k', ), color='k')

    ax.annotate('Zéro\ndes cartes', (1920, 0), (1925, -.3), size=8, ha='center', \
        arrowprops=dict(arrowstyle='-', connectionstyle='angle3,angleA=90,angleB=0') )


def plot_Sorel_annual_minimum_qom_levels():
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15930-1-JAN-1916_slev.csv')
    ts_min = GLSLio.annual_min_qom_ts(ra)

    # Graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.subplots_adjust(bottom=.1, right=.95, left=.1)
    ts_min.plot(ax=ax, color=c, lw=1.5)
    ts_min.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ylim = ax.get_ylim()
    dy = ylim[1] - ylim[0]

    ax.set_ylabel("Niveau [m]")
    ax.set_title("Minimum annuel du niveau d'eau à Sorel")
    for y in [2001, 2007, 2012]:
        ax.annotate(str(y), (y, ts_min[y]+dy*.02), (y, ts_min[y]+dy*.8), ha='center', size='small', arrowprops=dict(arrowstyle='-', fc='w', ec='k'), color='k')

    ax.text(1961, 0, 'Zéro des cartes', size=8, ha='left', )


def plot_Sorel_annual_minimum_qom_levels_flows():


    # Load levels
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15930-1-JAN-1916_slev.csv')
    level_min = GLSLio.annual_min_qom_ts(ra)





    # Load streamflow
    ts = GLSLio.Q_Sorel('qtm')
    g = ts.groupby(level=0)
    n = g.count()
    q_min = pd.TimeSeries(np.where(n>30, g.min(), np.nan), n.index)

    # Graph
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, facecolor='w')
    fig.subplots_adjust(bottom=.07, right=.95, left=.1, hspace=.1, top=.92)
    trans = [mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes) for ax in axes]

    level_min.plot(ax=axes[1], color=c, lw=1.5)
    level_min.plot(style='o', ax=axes[1], ms=5, mec=c, mfc='w', mew=1)

    q_min.plot(ax=axes[0], color=c, lw=1.5)
    q_min.plot(style='o', ax=axes[0], ms=5, mec=c, mfc='w', mew=1)

    axes[0].set_ylabel("Débit [m³/s]")
    axes[1].set_ylabel("Niveau d'eau [m]")
    axes[0].set_xlabel('')
    axes[1].set_xlabel("Année")
    axes[0].text(1, 1, "Minima annuels - Sorel", ha='right', va='bottom', size=24, color='#0A0C3D', weight='bold', alpha=.8, transform=axes[0].transAxes)

    for ax in axes:
        ax.grid(ls='-', lw=.1, color='#777777')
        ax.grid(axis='x')
        strip(ax)
        ax.set_ymargin(.07)
        ax.autoscale(enable=True, axis='y', tight=True)


    axes[1].set_xlim(right=2015)
    plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)

    for y in [2001, 2007, 2012]:
        axes[1].annotate(str(y), (y, level_min[y]), (y, .9),
            ha='center', size=11,
            textcoords=trans[1],
            arrowprops=dict(arrowstyle='-', fc='w', ec='k', shrinkB=5,), color='k')

    axes[1].text(1932, 0, 'Zéro des cartes', size=10, ha='left', )

    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="k", ec="k", lw=2)
    t = axes[0].text(1961, .01, "Régularisation des débits  ", ha="left", va="bottom",
            size=12, weight='bold', color='white', transform=trans[0],
            bbox=bbox_props)

    return fig, axes
    #plt.savefig('../figs/Sorel_q_l_min.png')
    #plt.savefig('../figs/Sorel_q_l_min.svg')


def plot_Sorel_frequential_analysis():

    hc = '#20344A'
    fc = '#F67D43'

    q = GLSLio.Q_Sorel('qtm')
    P, r = analysis.frequential_analysis(q)

    x = np.linspace(5000, np.exp(P.isf(.999)), 200)
    lx = np.log(x)

    fig, ax = plt.subplots(1,1,figsize=(8,5), facecolor='w')
    fig.subplots_adjust(bottom=.15, left=.15)

    y = P.pdf(lx)

    # Compute normalizing constant for the linear scale
    A = np.trapz(y, x)


    ax.plot(x, y/A, lw=2.5, color=hc, clip_on=False, label='Reconstitué')
    n, bins, patches = ax.hist(r.values, 20, normed=True, rwidth=.92, color='#C1CCD8', ec='none')

    sv = analysis.EC_scen_Q['Sorel'][:3]
    ax.plot(sv, P.pdf(np.log(sv))/A, 'o', ms=5, mec=hc, mfc='w', mew=1, clip_on=False)

    ax.annotate("1/10000 ans", (sv[0], P.pdf(np.log(sv[0]))/A), (-.1, -.15), textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle='-', connectionstyle='angle,angleA=-90,angleB=0,rad=5', fc='w', ec='k', lw=.3,), color=hc)
    ax.annotate("1/70 ans", (sv[1], P.pdf(np.log(sv[1]))/A), (.3, -.15), textcoords='axes fraction', ha='center', arrowprops=dict(arrowstyle='-', fc='w', ec='k', lw=.3,), color=hc)
    ax.annotate("1/3 ans", (sv[2], P.pdf(np.log(sv[2]))/A), (.49, .6), textcoords='axes fraction', ha='center', color=hc)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_position(('outward',10))
    #ax.set_xlabel('Débit à Sorel [m³/s]')
    ax.text(1.02, -.12, 'm³/s', ha='left', va='top', size='small', clip_on=False, transform=ax.transAxes)
    ax.xaxis.set_units('m³/s')
    ax.set_yticks([])
    ax.xaxis.tick_bottom()
    ax.annotate("Débits reconstitués à Sorel\nde 1932 à 2013", (bins[-4], n[-4]), (1, .9), textcoords='axes fraction', ha='right')

    # Same thing but minus 20%
    Pp, rp = analysis.frequential_analysis(q*.8)
    y = Pp.pdf(lx)
    A = np.trapz(y, x)
    ax.plot(x, y/A, lw=2.5, color=fc, clip_on=False, alpha=.7, label='Reconstitué moins 20%')
    ax.plot(sv, Pp.pdf(np.log(sv))/A, 'o', ms=5, mec=fc, mfc='w', mew=1, clip_on=False)

    plt.legend(loc='upper left', frameon=False, fontsize='small')
    ax.set_xlim(5000,10500)

    ax.annotate("1/75 ans", (sv[0], Pp.pdf(np.log(sv[0]))/A), (-18,10), textcoords='offset points', ha='center')
    ax.annotate("1/2 ans", (sv[1], Pp.pdf(np.log(sv[1]))/A), (0,10), textcoords='offset points', ha='center')

    plt.savefig('../figs/Sorel_frequential_analysis.svg')
    return ax



def plot_mesh():
    import matplotlib.transforms as mtransforms
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    cm = mpl.cm.spectral

    fig, ax = plt.subplots(1,1, figsize=(8,4))
    fig.set_facecolor('w')
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)


    AR = mtransforms.Affine2D()
    AR.rotate_deg(-25)
    pts = np.hstack(GLSLio.EC_pts().values())
    X, Y, Z = pts
    S = ax.scatter(X, Y, c=Z, s=.3, linewidths=0, transform=AR + ax.transData, cmap=cm)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.set_xlim(2355000, 2515000)
    ax.set_ylim(4420000, 4500000)
    ax.set_axis_off()

    axins = zoomed_inset_axes(ax, 10, loc=4) # zoom =
    axins.scatter(X, Y, c=Z, s=1, linewidths=0, transform=AR + axins.transData, cmap=cm)

    # sub region of the original image
    x1, y1 = 2459000, 4480000
    axins.set_xlim(x1, x1+5E3)
    axins.set_ylim(y1, y1+5E3)

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    axins.set_xticks([]); axins.set_yticks([])

    cbax = fig.add_axes([.1,.9,.45,.04])
    cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
    cb.set_label('Élévation [m]')

    data = []
    def onClick(event):
        data.append((event.xdata, event.ydata))

    def close(evt):
        print(data)
        return data

    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('close_event', close)

    return fig, ax


def plot_depth_map(scen, X=None, Y=None):
    from scipy.spatial import Delaunay
    import matplotlib.transforms as mtransforms
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset


    cm = haxby
    #cm.set_over('w')


    #m = basemap.Basemap(projection='tmerc', resolution='c', lat_0=45.8, lon_0=-73.5, k_0=0.99990, ellps='GRS80', width=170000, height=170000)
    proj = GLSLio.MTM8()
    #m.drawcoastlines()
    #m.drawcountries()
    #m.drawrivers()

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    cbax = fig.add_axes([.1,.9,.45,.04])
    fig.set_facecolor('w')
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)

    AR = mtransforms.Affine2D()
    AR.rotate_deg(-25)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.set_xlim(2355000, 2515000)
    ax.set_ylim(4420000, 4500000)
    ax.set_axis_off()

    # Plot control points
    cp = False
    if cp:
        for key, c in zip(analysis.CP['sites'], analysis.CP['coords']):
            x,y = proj(*c)
            plt.plot([x,], [y,], 'o', mfc='w', mec='b', ms=10, label=key, zorder=1)

        plt.legend()

    pts = np.hstack(GLSLio.EC_pts().values())
    D = np.hstack(GLSLio.EC_depth(scen=scen).values())

    X, Y, Z = pts
    D = np.ma.masked_less_equal(D, 0)
    S = ax.scatter(X, Y, c=D, s=.3, linewidths=0, transform=AR + ax.transData, cmap=cm, vmax=45, vmin=1e-2, norm=mpl.colors.LogNorm())



        # Convert coordinates into lat, long
        #lon, lat = proj(x, y, inverse=True)

        # Convert back into map projection
        #x, y  = m(lon, lat)

        # Construct Delaunay tesselation
        # I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
        #T = analysis.get_tesselation(reg)

        #plt.tricontourf(T, d, 20, mask=ma, vmin=0, vmax=20, cmap=cm)
        #plt.tripcolor(T, d, vmin=0, vmax=20, cmap=cm)

    cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
    cb.set_label('Profondeur [m]')
    #cb.formatter(ScalarFormatter(cb.ticks)

    axins = zoomed_inset_axes(ax, 3.5, loc=4, borderpad=0.3) # zoom =
    axins.scatter(X, Y, c=D, s=2, cmap=cm, vmax=45, vmin=1e-2, linewidths=0, transform=AR + axins.transData, norm=mpl.colors.LogNorm())

    # sub region of the original image
    x1, y1 = 2475000, 4482000
    axins.set_xlim(x1, x1+2.2E4)
    axins.set_ylim(y1, y1+1.1E4)

    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.7")
    axins.set_xticks([]); axins.set_yticks([])

    return fig, ax


def plot_map(bg=False):
    """Draw a nice map of the study region with the CCSM3 grid lines
    and the contour of a set of regions.

    Parameters
    ----------
    bg : bool
      Draw a nice color background
    regions : dict
      A dictionary of Region objects to draw.
    ccsm3 : bool
      Draw the CCSM3 atmospheric model grid.
    """


    # Create the projection and draw the state boundaries.
    fig = plt.figure(figsize=(8,4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111)
    map = basemap.Basemap(projection='stere',
                  lat_0=46., lon_0=-73.5, lat_ts=45, width=.6e6, height=.3e6,
                  resolution='i')
    map.drawstates(color='gray')
    map.drawcountries()
    #map.bluemarble()


    # Plot control points
    cp = True
    if cp:
        for name, c in zip(analysis.CP['names'], analysis.CP['coords']):
            x,y = map(*c)
            M = ax.plot([x,], [y,], 'o', mfc='w', mec='orange', ms=6, label=name, zorder=1, mew=2)
            ax.annotate(name, (x,y), (10, -3), textcoords='offset points')

    if bg:
        image = '/home/david/data/earth_image/NE1_HR_LC_SR_W_DR.tif'
        geo = dict(lon0=-180, lat0=90, dx=.016666666666, dy=-0.01666666666, x_aff=0, y_aff=0)
        lon, lat, bgim = crop_tiff(image, map.lonmin, map.latmin, map.lonmax, map.latmax, geo) # (3, ny, nx)

        im = np.empty((len(lat), len(lon), 3), np.uint8)
        for k in range(3):
            im[:,:,k] = map.transform_scalar(bgim[:, :, k], lon, lat[::-1], len(lon), len(lat))
        map.imshow(im, origin='upper')
    else:
        map.drawcoastlines()

    map.drawmapscale(-71., 45.0, -73.5, 46, 100, barstyle='fancy')

    return map

def crop_tiff(gtif, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, geo={}):
    import PIL


    I = PIL.Image.open(gtif)
    # lon0, lat0 corresponds to the node coordinate.
    # In the twf file, what is stored is the tracer coordinate.

    nx,ny = I.size
    lon0 = geo['lon0']
    lat0 = geo['lat0']
    dx = geo['dx']
    dy = geo['dy']
    x_aff = geo['x_aff']
    y_aff = geo['y_aff']


    lon1 = lon0 + nx*dx    + ny*x_aff
    lat1 = lat0 + nx*y_aff + ny*dy

    imin = int((llcrnrlon - lon0) / dx)
    imax = int((urcrnrlon - lon0) / dx)+1

    jmax = int((llcrnrlat - lat0) / dy)+1
    jmin = int((urcrnrlat - lat0) / dy)

    if x_aff == 0 and y_aff==0:
        lon = lon0 + dx/2.+ np.arange(imin, imax) * dx
        lat = lat0 + dy/2. + np.arange(jmin, jmax) * dy

    A = np.asarray(I.crop([imin, jmin, imax, jmax]))

    return lon, lat, A
