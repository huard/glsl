
import numpy as np
import scipy.stats
import pandas as pd
import GLSLio
from imp import reload
from matplotlib import pyplot as plt
reload(GLSLio)

gev = scipy.stats.genextreme

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
EC_scen_L = {'jetee': [4.29, 4.95, 5.61, 6.30, 7.19, 7.99, 8.8, 9.82],
             'varennes':[3.48, 4.2, 4.95, 5.57, 6.37, 7.2, 7.98, 9.06], 
             'sorel': [2.96, 3.56, 4.17, 4.74, 5.42, 6.22, 6.92, 8.01], 
             'tr': [2.52, 2.95, 3.55, 4.06, 4.69, 5.53, 6.16, 7.24]}

# Temps de retour
EC_scen_T = [10000, 70, 3, None, None, 2, 16, 7000]


def FanFay(site,):
    """Return the equation from Fan & Fay (2002) relating stage to 
    discharge at locations on the St. Lawrence:
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
     
     
    Notes
    -----
    Using the coefficients for the weighted series to compensate for 
    low level biases, but neglecting (for now) the QTM residuals and 
    the tidal component T. 
    """
    regcoefs = {'jetee':[(.001757, .000684, 0, 0.001161, 0.000483), 0.6587, 0.9392],
                'varennes':[(0.001438, 0.001377, 0, 0.001442, 0.000698), 0.6373, 1.0578],
                'sorel':[(0.001075, 0.001126, 0, 0.001854, 0.000882), 0.6331, 1.277],
                'lsp':[(0.000807, 0.001199, 0, 0.001954, 0.000976), 0.6259, 1.4722],
                'tr':[(.000584, .00069, .000957, .001197, .000787), .7042, 1.5895],
                #'tr':[(.000589, .000727, .00102, .001158, .000815), 0.6981, 1.5919],
                }
    """
    regcoefs = {'jetee': [(.001777, .000626, 0, .001173, .000532), .6575],
                'varennes': [], 
                'sorel': [],
                'lsp': [], 
                'tr': [(.000612, .000816, .001169, .001223, .0008666), .68], }
    """
    c, h, t = regcoefs[site.lower()]
    
    def func(Q, tidal):
        return np.dot(c, Q)**h + t * tidal
    return func
    
def FanFayLevel(site, scen='bc'):
    """Compute the level from flows."""
    qs = 'stl', 'dpmi', 'rich', 'fran', 'mau'
    Q = pd.DataFrame([GLSLio.FF_flow(q, scen) for q in qs])
    T = GLSLio.FF_tidal()
    
    f = FanFay(site)
    return f(Q, T)
    
    
def inferStMaurice(scen):
    """Find the St-Maurice streamflow making the level computed from the 
    Fan & Fay relationship at Trois-Rivières equal to that from the EC
    scenarios."""
    from scipy import optimize
    
    # Level from EC scenarios
    L = EC_scen_L['tr'][scen-1]
    
    # EC scenario streamflows
    qs = 'LaSalle', 'MIP', 'Richelieu', 'St-Francois'
    q = [EC_scen_Q[s][scen-1] for s in qs]
    
    # Fan & Fay relation at Trois-Rivières
    f = FanFay('tr')
    
    # St-Maurice streamflow such that the Fan & Fay level equals the EC scenario level.
    res = optimize.fmin(lambda x: np.abs(f(q+[x])-L), 695, disp=False)
    return res[0]
    
    
def FanFayInterpolation(lon, lat, scen):
    qs = 'LaSalle', 'MIP', 'Richelieu', 'St-Francois'

    # Get the streamflows
    q = [EC_scen_Q[s][scen-1] for s in qs] + [inferStMaurice(scen)]
    
    # Positions of the stations
    coords = {'jetee': (-73.5525, 45.5035),
              'varennes': (-73.443667, 45.684333),
              'sorel': (-73.115667, 46.047167),
              'lsp': (-72.8955, 46.194833),
              'tr': (-72.539167, 46.3405 ),
            }
    
    P = GLSLio.MTM8()
    
    x, y = P(lon, lat)
    
    dist = {}
    for key, c in coords.items():
        xc, yc = P(*c)
        dist[key] = np.hypot(x-xc, y-yc)
        
    i = np.argsort(dist.values())[:2]
    locs = dist.keys()[i]
    
    h = {}
    for l in locs:
        h[l] = FanFay(l)(q)
        
    
    
    
    
    
    
#def DischargeforFanFay():
#    q1 = GLSLio.get_dly("02OA016") + GLSLio.get_dly("02OA024")
#    q2 = GLSLio.get_dly("02OA004") + GLSLio.get_dly("02OA003")
#    q3 = GLSLio.get_dly("02OF019")
#    q4 = GLSLio.get_dly("02NG005")   
#    return q1, q2, q3, q4
    

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
    
    
    
    
    params = scipy.stats.pearson3.fit(y, -.2)
    P = scipy.stats.pearson3(*params)
    return P
    print (1/G.sf(5000))
    
    plt.hist(am.values, 20, normed=False)
    x = np.linspace(am.min()-100, am.max()+100, 100)
    y = np.log(x)
    plt.plot(x, G.pdf(y), label='Gamma')
    plt.plot(x, P.pdf(y), label='Pearson III')
    plt.legend()
    return P
    
def pearson_mom(x):
    
    m1 = np.mean(x)
    m2 = np.var(x)
    C = scipy.stats.skew(x)
    
    return scipy.stats.pearson3(scipy.stats.skew(x), loc=np.mean(x), scale=np.std(x))
    
    
    #beta = (2/C)**2   # shape
    #alpha = np.sqrt(m2/beta) # scale
    #gamma = m1 - np.sqrt(m2*beta) # loc
    
    
def logp_at_Lasalle():
    """Frequency analysis of streamflow at LaSalle."""
    scipy.stats
    

    
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
            l = FanFay(site)(q)
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
    
