import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import basemap

import GLSLio
import analysis
from imp import reload
reload(analysis)

mpl.rcParams['svg.fonttype'] = 'none'
c = '#00425C'


    
    
    
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
    q_min = ts.groupby(level=0).min()
    
    # Graph
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(bottom=.07, right=.95, left=.1, hspace=.03)
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
    
    axes[1].set_xlim(right=2015)
    
    for y in [2001, 2007, 2012]:
        axes[1].annotate(str(y), (y, level_min[y]), (y, .9), 
            ha='center', size=11, 
            textcoords=trans[1],
            arrowprops=dict(arrowstyle='-', fc='w', ec='k', shrinkB=5,), color='k')

    axes[1].text(1932, 0, 'Zéro des cartes', size=10, ha='left', )

    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="k", ec="k", lw=2)
    t = axes[0].text(1961, .05, "Régularisation des débits  ", ha="left", va="bottom", 
            size=12, weight='bold', color='white', transform=trans[0],
            bbox=bbox_props)

    plt.savefig('../figs/Sorel_q_l_min.png')
    plt.savefig('../figs/Sorel_q_l_min.svg')


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
    
def triangle_area(x,y,i):
    """Return area of triangles formed by vertices (x[i], y[i]).
    
    Parameters
    ----------
    x,y: ndarray (n,)
      xy coordinates
    i : ndarray (n,3)
      Indices of triangle vertices.
    
    Return
    ------
    out : ndarray (n,)
      Triangle area.
    """
    n, d = np.asarray(i).shape
    assert d == 3
    
    a = np.zeros(n)
    for k in range(3):
        a += x[i[:,k]] * (y[i[:,(k+1)%3]] - y[i[:,(k+2)%3]])
    
    return np.abs(a)
     
    
      
def Tesselation(domain):
    """Return the model tesselation in lat, lon coordinates.
    
    Parameters
    ----------
    domain : {'lsl', 'lsp', 'mtl_lano'}
      Domain name.
      
    """
    from scipy.spatial import Delaunay
    # Get grid coordinates
    x, y = GLSLio.get_scen(1, domain, ('X', 'Y'))
        
    # Convert coordinates into lat, long
    # proj = GLSLio.MTM8()
    # lon, lat = proj(x, y, inverse=True)
        
    # Construct Delaunay tesselation
    # I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
    D = Delaunay(np.array([x,y]).T)
    T = tri.Triangulation(x, y, D.vertices)
    
    # Masked values
    area = triangle_area(x, y, D.vertices)
    dist = np.max(np.sqrt(np.diff(x[D.vertices], axis=1)**2 + np.diff(y[D.vertices])**2), axis=1)
    
    ma = (area > 1e6) | (dist > 1e3)
    T.set_mask(ma)
    return T
    

def plot_depth_map(scen):
    from scipy.spatial import Delaunay
    
    cm = mpl.cm.gist_ncar
    #cm.set_under('w')
    
    m = basemap.Basemap(projection='tmerc', resolution='c', lat_0=45.8, lon_0=-73.5, k_0=0.99990, ellps='GRS80', width=170000, height=170000)
    proj = GLSLio.MTM8()
    #m.drawcoastlines()
    m.drawcountries()
    #m.drawrivers()
    
    for reg in 'lsl', 'lsp', 'mtl_lano':
        x, y, z, d, v = GLSLio.get_scen(scen, reg)
        
        # Convert coordinates into lat, long
        lon, lat = proj(x, y, inverse=True)
        
        # Convert back into map projection
        x, y  = m(lon, lat)
        
        # Construct Delaunay tesselation
        # I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
        D = Delaunay(np.array([x,y]).T)
        T = tri.Triangulation(x, y, D.vertices)
        
        
        # Masked values
        area = triangle_area(x, y, D.vertices)
        dist = np.max(np.sqrt(np.diff(x[D.vertices], axis=1)**2 + np.diff(y[D.vertices])**2), axis=1)
        #md = d[T.triangles].min(axis=1) 
        ma = (area > 1e6) | (dist > 1e3)
        T.set_mask(ma)
        #print (np.sum(ma))
        
        
        #plt.tricontourf(T, d, 20, mask=ma, vmin=0, vmax=20, cmap=cm)
        plt.tripcolor(T, d, vmin=0, vmax=20, cmap=cm)
    
    cb = plt.colorbar()
    cb.set_label('Profondeur [m]')
    
    
    
    
    
    return
    
    
    
