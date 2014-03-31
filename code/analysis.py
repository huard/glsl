import numpy as np
import scipy.stats
import pandas as pd
import GLSLio
from imp import reload
reload(GLSLio)

gev = scipy.stats.genextreme

def gev_at_Lasalle():
    
    ts = GLSLio.get_station('02OA016')
    gr = ts.groupby(lambda x: x.year)
    am = gr.min()
    n = gr.count()
    am[n<300] = np.nan
    
    return am
    
    
def Q_Sorel():
    """
    Le débit reconstitué à Sorel est composé des débits suivants:
    
    Mille-Iles (02OA003)
    Prairies (02OA004)
    St-Laurent@LaSalle (02OA016)
    Canal Rive Sud (02OA024) + (0.146*Chateauguay) + (4.288*Delisle)
    Richelieu (02OJ007) + 0.493*Yamaska
    Assomption (02OB008) + 0.292*Chateauguay + 5.306*Delisle + 0.799*Assomption + 1.153*duNord + 0.075*Yamaska
    
    
    
    Chateauguay (02OA054)
    Delisle (02MC028)
    Yamaska (02OG043)
    Assomption (02OB008
    """
    Q = GLSLio.get_dly('02OA003', 'Q') + \
    GLSLio.get_dly('02OA004', 'Q') + \
    LaSalle_REC() + \
    GLSLio.get_dly('02OA024', 'Q').fillna() + \
    GLSLio.get_dly('
    
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
    
