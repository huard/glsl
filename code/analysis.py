
import numpy as np
import scipy.stats
import pandas as pd
import GLSLio
from imp import reload
from matplotlib import pyplot as plt
reload(GLSLio)

gev = scipy.stats.genextreme

"""
Fan & Fay: Common 29 years time period: 1962-1990
Base case (bc) is meteorological data driven through the GL model, including regulation.
WW and WD cases are the base case to which a delta is applied.
"""


"""Variables tirées du rapport de Bouchard et Morin."""
# Débits des scénarios à Sorel
EC_scen_Q = {'Sorel':[5000, 6500, 8000, 9500, 12000, 14500, 17500, 20500],
             'LaSalle':[4572, 5740, 6997, 8304, 10102, 11396, 13174, 14531],
             'MIP':[398, 728, 960, 1142, 1750, 2772, 3824, 5374],
             'Assomption':[30,32,43,54,148, 332, 502, 550],
             'Richelieu':[137, 148, 240, 326, 615, 898, 1044, 1100],
             'Yamaska': [28, 29, 38, 52, 126, 220, 345, 410],
             'St-Francois':[120, 128, 139, 155, 330, 572, 850, 980],
             'Trois-Rivieres':[5319, 6843, 8469, 10093, 13227, 16517, 20188, 23554],
            }

# Niveaux printemps IGLD85
EC_scen_L = {'mtl': [4.29, 4.95, 5.61, 6.30, 7.19, 7.99, 8.8, 9.82],
             'var':[3.48, 4.2, 4.95, 5.57, 6.37, 7.2, 7.98, 9.06],
             'srl': [2.96, 3.56, 4.17, 4.74, 5.42, 6.22, 6.92, 8.01],
             'trois': [2.52, 2.95, 3.55, 4.06, 4.69, 5.53, 6.16, 7.24]}

# Temps de retour
EC_scen_T = [10000, 70, 3, None, None, 2, 16, 7000]
""" --- """


# Points de contrôle (F&F) - Positions des stations de niveau d'EC
CP = dict(sites = ('mtl', 'var', 'srl', 'lsp', 'trois'),
          coords = ((-73.5525, 45.5035), (-73.443667, 45.684333), (-73.115667, 46.047167), (-72.8955, 46.194833), (-72.539167, 46.3405 )),
          names = ('Jetée #1', 'Varennes', 'Sorel', 'Lac St-Pierre', 'Trois-Rivières'))



def stage_discharge_ff(site,):
    """Return the equation from Fan & Fay (2002) relating stage to
    discharge at locations on the St. Lawrence.

    Parameters
    ----------
    site : {'mtl', 'var', 'srl', 'lsl', trois'}
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

def get_levels_ff(site, scen='bc'):
    """Return the levels from the Fan & Fay scenarios.

    Parameters
    ----------
    site : {'mtl', 'var', 'srl', 'lsl', trois'}
      Name of control point.

    scen : {'bc', 'wd', 'ww'}
      Scenario name: Base Case, Warm & Dry, Warm & Wet.

    """
    qs = 'stl', 'dpmi', 'rich', 'fran', 'mau'
    Q = pd.DataFrame([GLSLio.FF_flow(q, scen) for q in qs])

    K = GLSLio.FF_K(site)

    T = GLSLio.FF_tidal()

    f = stage_discharge_ff(site)
    return f(K*Q, T)

def get_flow_sorel(scen='bc'):
    """Sum the flow from tributaries to get the flow at Sorel from the
    F&F flows."""
    qs = 'stl', 'dpmi', 'chat', # Is there something else, do we need to scale those ?
    Q = [GLSLio.FF_flow(q, scen) for q in qs]
    return sum(Q)


def get_tesselation(domain):
    """Return the model tesselation in native coordinates.

    Parameters
    ----------
    domain : {'lsl', 'lsp', 'mtl_lano'}
      Domain name.

    """
    from scipy.spatial import Delaunay
    import matplotlib.tri as tri

    # Get grid coordinates
    x, y = GLSLio.get_scen(1, domain, ('X', 'Y'))

    # Convert coordinates into lat, long
    # proj = GLSLio.MTM8()
    # lon, lat = proj(x, y, inverse=True)

    # Construct Delaunay tesselation
    # I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
    #D = Delaunay(np.array([x,y]).T)
    T = tri.Triangulation(x, y)# D.vertices)

    #TODO:  Use circle-ratios instead
    A = tri.TriAnalyzer(T)
    ma = A.get_flat_tri_mask(.2)
    # Masked values
    #area = _triangle_area(x, y, T.triangles)
    dist = np.max(np.sqrt(np.diff(x[T.triangles], axis=1)**2 + np.diff(y[T.triangles])**2), axis=1)
    ma = ma | (dist > 200)

    T.set_mask(ma)

    return T

def _triangle_area(x,y,i):
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

def save_convex_hulls():
    """For all three grids, save the convex hull in a json file."""
    import json
    from scipy.spatial import ConvexHull
    qhull = {}
    for dom in ['lsl', 'lsp', 'mtl_lano']:
        x, y = GLSLio.get_scen(8, dom, ('X', 'Y'))
        pts = np.array([x,y]).T
        qh = ConvexHull( pts )
        qhull[dom] = pts[qh.vertices].tolist()
        #T = get_tesselation(dom)
        #qhull[dom] = np.array([T.x, T.y]).T[T.convex_hull].reshape(-1,2).tolist()


    with open('../analysis/qhulls.json', 'w') as f:
        json.dump(qhull, f)


def get_domain(lon, lat):
    """Return the domain holding the coordinates."""
    import json
    import matplotlib as mpl

    # Convert coordinates in native grid coordinates
    P = GLSLio.MTM8()
    x, y = P(lon, lat)

    # Load convex_hulls
    qhulls = json.load(open('../analysis/qhulls.json'))

    for key, poly in qhulls.items():
        P = mpl.path.Path(poly)
        if P.contains_point((x,y)):
            return key

    raise ValueError("Point not in any domain")


def interpolate_EC_levels(lon, lat):
    """Return the eight levels interpolated from the grid data."""
    from matplotlib import tri
    # Native coordinates
    P = GLSLio.MTM8()
    x, y = P(lon, lat)

    # Domain identification
    dom = get_domain(lon, lat)

    # Interpolate from grid
    X, Y = GLSLio.get_scen(8, dom, variables=('X', 'Y'))
    pts = np.array([X,Y]).T

    # Get bottom level
    Z = GLSLio.get_scen(1, dom, variables=('Z',)).T
    I = scipy.interpolate.LinearNDInterpolator(pts, Z)
    z = I(x, y)[0]

    # Compute depth for each scenario
    out = []
    for scen in range(1,9):
        D = GLSLio.get_scen(scen, dom, variables=('PROFONDEUR',)).T
        I.values = D
        out.append( I(x, y)[0] )

    # Add depth to bottom level
    return np.array(out) + z

def save_control_points_EC_levels():
    levels = []
    for c in CP['coords']:
        try:
            levels.append( interpolate_EC_levels(*c) )
        except:
            levels.append(None)
    return levels

def interpolate_ff_levels(lon, lat, scen='bc'):
    """For a given point, compute the value interpolated from the Fan &
    Fay levels at the control points (Mtl, Sorel, Varennes and
    Trois-Rivieres) and improve the interpolation using the 2D model
    results.

    Parameters
    ----------
    lon, lat : floats
      Geographic coordinates of point of interest.
    scen : {'bc', 'wd', 'ww'}
      Climate change scenario: base case, warm & dray, warm & wet.

    Returns
    -------
    out : ndarray
      Series of levels interpolated from Fan & Fay levels and corrected
      using the 2D simulation from EC.


    Notes
    -----
    1. Interpolate linearly the Fan & Fay levels
    1.1. Find the upstream and downstream control points
    1.2 Compute the weights to apply to upstream and downstream levels
    1.3 Compute the upstream and downstream levels from the F&F relations
    1.4 Interpolate at the chosen location based on the distance to the control points

    2. Correct for non-linear levels
    2.1 Compute the streamflow at Sorel
    2.2 Identify the scenarios above and below the Sorel streamflow
    2.3 Interpolate the scenario level at the point of interest
    2.4 Get the levels at the upstream and downstream control points for all 8 scenarios and interpolate at the point of interest
    2.5 Compute the difference between the linearly interpolated values between the control points and the interpolated 2D depth.
    2.6 Apply the difference to the Fan & Fay series interpolated at the point of interest

    """
    # Model domain
    dom = get_domain(lon, lat)

    # 1. Fan & Fay linear interpolation
    # ---------------------------------

    # 1.1 Get the upstream and downstream stations from point
    P = GLSLio.MTM8()
    x, y = P(lon, lat)

    dist = []
    for c in CP['coords']:
        xc, yc = P(*c)
        dist.append( np.hypot(x-xc, y-yc) )

    dist = np.array(dist)
    si = np.argsort(dist)[:2]

    upstream = CP['sites'][si.min()]
    downstream = CP['sites'][si.max()]
    #print( upstream, downstream )

    # 1.2 Compute the weights to apply to upstream and downstream levels
    Iw = dist[si.max()] / np.sum(dist[si])

    # 1.3 Compute the upstream and downstream levels from the F&F relations
    upL = get_levels_ff(upstream, scen)
    dnL = get_levels_ff(downstream, scen)

    # 1.4 Interpolate at the chosen location based on the distance to the control points
    FF_L = Iw * upL + (1-Iw) * dnL

    # 2. Correct for non-linear levels
    # --------------------------------

    # 2.1 Compute streamflow at Sorel
    FS = get_flow_sorel(scen)

    # 2.2 Identify the scenarios above and below the Sorel streamflow
    refQ = EC_scen_Q['Sorel']
    wi = np.interp(FS, refQ, range(1,9))
    wi[FS.isnull().values] = 0

    # 2.3 Interpolate the scenario level at the point of interest
    L = interpolate_EC_levels(lon, lat)

    # 2.4 Get the levels at the upstream and downstream control points for all 8 scenarios and interpolate at the point of interest
    EC_L = Iw * np.array(EC_scen_L[upstream]) + (1-Iw) * np.array(EC_scen_L[downstream])

    # 2.5 Compute the difference between the linearly interpolated values between the control points and the interpolated 2D depth.
    deltas = EC_L - L

    # 2.6 Apply the difference to the Fan & Fay series interpolated at the point of interest
    return FF_L + weight_EC_levels(deltas, wi)



def weight_EC_levels(levels, wi):
    """Interpolate within the 8 EC levels.

    Parameters
    ----------
    site : {'mtl', 'var', 'srl', 'trois'}
      Control point: Montréal, Varennes, Sorel and Trois-Rivières
    wi : ndarray
      Weighted index. For example, 3.7 means that EC scenario 3 has weight .3 and
      scenario 4 has weight .7.

    Returns
    -------
    out : ndarray
      The levels at sites interpolated from the 8 EC levels according to wi.
    """
    levels = np.asarray(levels)
    w1 = (np.floor(wi) - 1).astype(int)
    w = 1.-np.mod(wi,1)

    out = levels[w1] * w + levels[w1+1]*(1-w)
    out[w1<0] = np.nan
    return out



def frequential_analysis(ts):
    """
    Perform a frequential analysis of the annual minima using a
    Log-Pearson type III distribution.

    The LP3 distribution is fitted by the method of L-Moments, using
    Hosking's code.
    """

    import bayesidf

    # Type checking and annual minimum finding
    if type(ts.index) == pd.MultiIndex:
        assert ts.index.names[0] == 'Year'
        am = ts.groupby(level=0).min()
    elif type(ts.index) == pd.DateTimeIndex:
        assert ts.index.freq == pd.tseries.offsets.Day
        gr = ts.groupby(lambda x: x.year)
        am = gr.min()
        n = gr.count()
        am[n<300] = np.nan

    # Frequential analysis
    y = np.log(am.dropna())
    lm = bayesidf.samlmu(y)
    p = bayesidf.pelpe3(lm)
    mu, sigma, gamma = p

    return scipy.stats.pearson3(gamma, mu, sigma), am.dropna()



def Chateauguay_REC(validation=False):
    """
    Débits à la station 054 de 1970 à aujourd'hui et débits
    à la station 001 mis-à-l'échelle pour la période pré-1970.
    """

    # Main series
    q = GLSLio.get_dly('02OA054', 'Q')

    # Secondary series
    qp = GLSLio.get_dly('02OA001', 'Q') * 1.012

    # Fill missing values in main using secondary series
    q, qp = q.align(qp)
    nu = q.isnull()
    q[nu] = qp[nu]

    return q[nu].fillna()

def Yamaska_REC(validation=False):
     pass

def check():
    sites = ['jetee', 'varennes', 'sorel']
    qs = 'LaSalle', 'MIP', 'Richelieu', 'St-Francois'

    for site in ['Scen'] + sites:
            print(site, end=' \t' )
    print('')

    for scen in range(1,9):
        print (scen, end=' \t')
        q = [EC_scen_Q[s][scen-1] for s in qs] + [inferStMaurice(scen)]
        for site in sites:
            l = stage_discharge_ff(site)(q)
            print (np.around(l - EC_scen_L[site][scen-1],4), end=' \t')
        print('')


def LaSalle_REC(validation=False):
    """Débits disponibles à partir de 1955.
    Niveaux disponibles entre 1932 et 1978.

    Les débits pour la période 1932-1955 produits avec la relation

        Q = A*(WL-C)^B

    où A=95.0586498724
       B=2.5689817270
       C=-4.5575002237
       WL = Niveau d’eau en datum local
    """

    A=95.0586498724
    B=2.5689817270
    C=-4.5575002237

    q = GLSLio.get_dly('02OA016', 'Q')
    h = GLSLio.get_dly('02OA016', 'H')

    h1 = h[:q.index[0]][:-1]

    qh = A*(h1 - C)**B

    # Compare values over the period where they overlap
    if validation:
        pass
        #TODO

    return pd.concat((qh, q))
