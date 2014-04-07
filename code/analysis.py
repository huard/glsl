
import numpy as np
import scipy.stats
import pandas as pd
import GLSLio
from imp import reload
from matplotlib import pyplot as plt
reload(GLSLio)

gev = scipy.stats.genextreme

def FanFay(site,):
    """Return the equation from Fan & Fay (2002) relating stage to 
    discharge at locations on the St. Lawrence:
     * Jetée #1
     * Varennes
     * Sorel 
     * Lac St-Pierre
     
    The function will then accept a vector of flows from 
     * Lac St-Louis (02OA016, 02OA024)
     * Des Prairies & Milles-Iles (02OA004, 02OA003)
     * St-François (02OF019)
     * St-Maurice (02NG005)
     
     
    Notes
    -----
    Using the coefficients for the weighted series to compensate for 
    low level biases, but neglecting (for now) the QTM residuals and 
    the tidal component T. 
    """
    regcoefs = {'j':[(.001757, .000684, 0.001161, 0.000483), 0.6587],
                'v':[(0.001438, 0.001377, 0.001442, 0.000698), 0.6373],
                's':[(0.001075, 0.001126, 0.001854, 0.000882), 0.6331],
                'l':[(0.000807, 0.001199, 0.001954, 0.000976), 0.6259]}
                 
    c, h = regcoefs[site[0].lower()]
    
    return lambda Q: np.dot(c, Q)**h
    
def DischargeforFanFay():
    q1 = GLSLio.get_dly("02OA016") + GLSLio.get_dly("02OA024")
    q2 = GLSLio.get_dly("02OA004") + GLSLio.get_dly("02OA003")
    q3 = GLSLio.get_dly("02OF019")
    q4 = GLSLio.get_dly("02NG005")
    
    return q1, q2, q3, q4
    

def frequential_analysis(ts):
    """
    ts = GLSLio.get_dly('02OA016')
    """
    """
    gr = ts.groupby(lambda x: x.year)
    am = gr.min()
    n = gr.count()
    am[n<300] = np.nan
    """
    am = ts.groupby(level=0).min()
    plt.hist(am.values, 20, normed=False)
    
    
    y = np.log(am.dropna())
    
    params = scipy.stats.gamma.fit(y)
    G = scipy.stats.gamma(*params)
    
    
    params = scipy.stats.pearson3.fit(y, -.2)
    P = scipy.stats.pearson3(*params)
    return P
    print (1/G.sf(5000))
    
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
    
