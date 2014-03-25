import numpy as np
import scipy.stats

gev = scipy.stats.genextreme

def gev_at_Lasalle():
    import GLSLio
    ts = GLSLio.get_station('02OA016')
    gr = ts.groupby(lambda x: x.year)
    am = gr.min()
    n = gr.count()
    am[n<300] = np.nan
    
    return am
    
    
