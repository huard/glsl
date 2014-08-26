import graphs, GLSLio, FFio, analysis, GLSLutils, HYDATio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os, pickle, json
from imp import reload

reload(graphs)
reload(analysis)


def station_level_stats():
  for key, sid in GLSLio.stations_niveaux.items():
    fig, ax = graphs.plot_station_qom_level_stats(sid)
    fig.savefig('../rapport/figs/stats_niveaux_{0}.png'.format(key))
    plt.close()
#
def scenario_depths():
    for scen in range(1,9):
        fig, ax = graphs.plot_depth_map(scen)
        fig.savefig('../rapport/figs/depth_scen_{0}.png'.format(scen), dpi=200)
        plt.close()
#
def FF_scenarios():
    for site in ['ont']:
        fig, ax = graphs.plot_FF_flow(site)
        fig.savefig('../rapport/figs/FF_scenarios_{0}.png'.format(site), dpi=200)
        plt.close()
#
def plot_mesh():
    fig, ax = graphs.plot_mesh()
    fig.savefig('../rapport/figs/mesh.png', dpi=200)

def plot_Sorel():
    fig, axes = graphs.plot_Sorel_annual_minimum_qom_levels_flows()
    fig.savefig('../rapport/figs/Sorel_flow_levels.png', dpi=200)
    plt.close()
#

## Senario #1 for Pointe-Claire
def AECOM():
    """Notes: le scénario livré en juillet a été produit avec un y0=1965 au lieu de 1975. L'erreur sur la correction est au pire de moins de 14%.
    Ça mérite d'envoyer une version corrigée.
    """
    out = {}
    out['WI1'] = {}
    out['WI2'] = {}

    # Observations at Pointe-Claire
    s = '02OA039'
    L = HYDATio.get_hydat(s, 'H')
    M = HYDATio.get_station_meta(s)
    lat, lon = M['LATITUDE'], M['LONGITUDE']

    LQ = GLSLutils.group_qom(L).mean()
    LQ.index.names = ["Year", "QTM"]
    F = LQ.to_frame('Level m')
    out['obs'] = F
    F = F.swaplevel(0,1)

    fn = 'Water_Levels_OBS_Pointe-Claire_02OA039_IGLD85.xlsx'
    F.to_panel().to_excel(os.path.join('..', 'deliverables', 'AECOM', fn))


    # What-if scenario #1
    for scen in ['bc', 'wd']:
        l = FFio.PCL(scen)
        q = FFio.FF_flow('stl', scen)
        fn = 'Water_Levels_WhatIf1_Pointe-Claire_{0}_IGLD85.xlsx'.format(scen.upper())
        fr = dump_xls(q.reindex(q.index.truncate(after=29)), l.reindex(l.index.truncate(after=29)), os.path.join('..', 'deliverables', 'AECOM', fn), FFio.offset[scen])
        out['WI1'][scen.upper()] = fr

    # What-if scenario #2 at Sorel
    # See http://www.ngs.noaa.gov/PUBS_LIB/NAVD88/navd88report.htm
    # NAV88 and IGLD85 seem to be one and the same...
    ECL = analysis.interpolate_EC_levels(lon, lat)
    for scen, ts in zip(("BC", "SC"), analysis.scenario_2()):

        # Scenario weights
        wi = analysis.get_EC_scenario_index(ts)

        # Compute flow at Lasalle
        tsLS = analysis.weight_EC_levels(analysis.EC_scen_Q['LaSalle'], wi)

        # Compute level at LaSalle
        lLS = analysis.weight_EC_levels(ECL, wi)

        fn = 'Water_Levels_WhatIf2_Pointe-Claire_{0}_IGLD85.xlsx'.format(scen)
        fr = dump_xls(pd.Series(tsLS, ts.index), pd.Series(lLS, ts.index), os.path.join('..', 'deliverables', 'AECOM', fn))
        out['WI2'][scen] = fr

    pickle.dump(out, open('../analysis/aecom.pickle', 'wb'))

def dump_xls(flow, level, filename, offset=0):
    Fr = level.valid().to_frame('Level m')
    Fr['Flow m3s'] = flow
    y, q = Fr.index.levels
    Fr.index.set_levels((y+offset, q), True)
    out = Fr

    Fr = Fr.swaplevel(0,1)

    P = Fr.to_panel()
    P.to_excel(filename)
    return out


def ECOSYSTEMES():
    # Datum ID: 431, Great Lakes International Datum 1985
    # From EC datum reference:
    # Station	Nom de la station	RN de référence	Altitude(ZC)	Altitude(SRIGL85)	SRIGL85(ZC)
    # 15975      Lac Saint-Pierre	 87L9000	   	 6.537	       9.980        	    3.443

    # Les données du modèles sont en Niveau Moyen des Mers, ce qui je pense,
    # correspond à l'altitude. Si on veut reporter ces valeurs au datum de la
    # station de mesure, on doit donc soustraire 6.537 des scénarios.

    # Level observations at station
    # 02OC016    SAINT-PIERRE (LAC) AU CURB NO. 2
    # Stations H Lac St-Pierre
    out = {}
    out['WI1'] = {}
    out['WI2'] = {}

    s = '02OC016'

    L = HYDATio.get_hydat(s, 'H')
    M = HYDATio.get_station_meta(s)
    lat, lon = M['LATITUDE'], M['LONGITUDE']

    LQ = GLSLutils.group_qom(L).mean()
    LQ.index.names = ["Year", "QTM"]
    F = LQ.to_frame('Level m')
    out['obs'] = F
    F = F.swaplevel(0,1)

    fn = 'Water_Levels_OBS_Lac_St-Pierre_02OC016_IGLD85.xlsx'
    F.to_panel().to_excel(os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))

    # What-if scenario #1
    for scen in ['bc', 'wd']:
        l = FFio.level_series_QH('lsp', scen)
        q = FFio.total_flow('lsp', scen)
        fn = 'Water_Levels_WhatIf1_Lac_St-Pierre_{0}_IGLD85.xlsx'.format(scen.upper())
        fr = dump_xls(q.reindex(q.index.truncate(after=FFio.offset[scen]+29)), l.reindex(l.index.truncate(after=FFio.offset[scen]+29)), os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))
        out['WI1'][scen.upper()] = fr

    # What-if Scenario #2
    ECL = analysis.interpolate_EC_levels(lon, lat)
    for scen, ts in zip(("BC", "SC"), analysis.scenario_2()):

        # Scenario weights
        wi = analysis.get_EC_scenario_index(ts)

        # Compute flow at Trois-Rivières (Est-ce qu'on enlève le débit de la Nicolet? p 48 Morin, Bouchard)
        tsLSP = analysis.weight_EC_levels(analysis.EC_scen_Q['Trois-Rivieres'], wi)

        # Compute level at Lac St-Pierre
        lLSP = analysis.weight_EC_levels(ECL, wi)

        fn = 'Water_Levels_WhatIf2_Lac_St-Pierre_{0}_IGLD85.xlsx'.format(scen)
        fr = dump_xls(pd.Series(tsLSP, ts.index), pd.Series(lLSP, ts.index), os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))
        out['WI2'][scen] = fr

    pickle.dump(out, open('../analysis/ecosystemes.pickle', 'wb'))
    return out

#
def TOURISME():
    """Certains des positions des marinas correspondent aux batiments, et non au bassin de la marina.
    J'ai contacté Stéphanie pour qu'on décide ce qu'on fait avec ca."""
    meta = GLSLio.marinas()
    
    # Interpolate the levels at the marinas
    ECLpath = '../analysis/marinasECL_N.json'
    if os.path.exists(ECLpath):
        ECL = json.load(open(ECLpath))
    else:
    
        ECL = {}
        for k, v in meta.items():
            try:
                ECL[k] = analysis.interpolate_EC_levels(*v).tolist()
            except ValueError:
                ECL[k] = None
        json.dump(ECL, open(ECLpath, 'w'))
        
    out = {'REF':{} ,'WI1':{}, 'WI2':{}}
    
    # Scenario #1
    for (key, L) in ECL.items():
        if L is not None:
            lon, lat = meta[key]
            out['WI1'][key] = analysis.interpolate_ff_levels(lon, lat, 'wd', L)
            
    # Scenario #2
    for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):
        for (key, L) in ECL.items():
            if L is not None:
                # Scenario weights
                wi = analysis.get_EC_scenario_index(ts)
    
                # Compute levels
                out[scen][key] = pd.Series(analysis.weight_EC_levels(L, wi), ts.index)

    for (key, L) in ECL.items():
        if L is not None:
            fn = '../deliverables/Tourisme/Scenarios_niveaux_{0}_IGLD85.xlsx'.format(key)
            EW = pd.ExcelWriter(fn)
            for i, scen in enumerate(['REF', 'WI1', 'WI2']):
                out[scen][key].to_frame(scen + ' m').to_excel(EW, scen)
            EW.close()

    return out
            
    
    

def FONCIER():
    """Niveaux au Lac St-Louis (Pointe-Claire)"""
    fn = '../deliverables/Scenarios_niveaux_Pointe-Claire_02OA039.xlsx'
    EW = pd.ExcelWriter(fn)
    out = {}

    # Observations at Pointe-Claire
    s = '02OA039'

    M = HYDATio.get_station_meta(s)
    lat, lon = M['LATITUDE'], M['LONGITUDE']

    if 0: #OBS
        L = HYDATio.get_hydat(s, 'H')
        LQ = GLSLutils.group_qom(L).mean()
        LQ.index.names = ["Year", "QTM"]
        F = LQ.to_frame('Level m')
        out['obs'] = F

    # What-if scenario #1
    wd = FFio.PCL('wd')
    out['WI1'] = wd.reindex(wd.index.truncate(after=2068))

    # What-if scenario #2 at Sorel
    ECL = analysis.interpolate_EC_levels(lon, lat)
    for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):

        # Scenario weights
        wi = analysis.get_EC_scenario_index(ts)

        # Compute level at Pointe-Claire
        out[scen] = pd.Series(analysis.weight_EC_levels(ECL, wi), ts.index)

    for i, scen in enumerate(['REF', 'WI1', 'WI2']):
        out[scen].to_frame(scen + ' m').to_excel(EW, scen)
    EW.close()

    return out



def HYDRO():
    """Scénario de débit pour Beauharnois et Les Cedres."""

    def do(site):
        fn = '../deliverables/Scenarios_debit_lac_Ontario.xlsx'
        EW = pd.ExcelWriter(fn)
        tmp = {}
    
        # 1
        wd = FFio.FF_flow('ont', 'wd')
        tmp['WI1'] = wd.reindex(wd.index.truncate(after=2069))
    
        #2
        qbc, q2 = analysis.scenario_2()
    
        ECQ = np.array( analysis.EC_scen_Q['Beauharnois'] ) + np.array( analysis.EC_scen_Q['lesCedres'] )
    
        # Scenario weights
        for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):
            wi = analysis.get_EC_scenario_index(ts)
    
            # Compute flow at LaSalle
            tmp[scen] = pd.Series(analysis.weight_EC_levels(ECQ, wi), ts.index)
    
    
        for i, scen in enumerate(['REF', 'WI1', 'WI2']):
            tmp[scen].to_frame(scen + ' m3s').to_excel(EW, scen)
        EW.close()
        return tmp

#
def plot_ww_wd_scenarios():
    fig, ax = graphs.plot_FF_flow('ont')
    fig.savefig('../figs/FF_wd_ww_scenarios_ontario.png')
    plt.close()

#
def plot_aecom():
    from matplotlib.transforms import offset_copy
    data = pickle.load(open('../analysis/aecom.pickle', 'rb'))
    fig = graphs.scenarios(data)

    fig.axes[0].text(0, 1, "Pointe-Claire", fontsize=34, weight='bold', color='#2b2929', alpha=.8, transform=offset_copy(fig.axes[0].transAxes, x=0, y=30, units='dots'))
    fig.savefig('../figs/Pointe-Claire.png')

#
def plot_ecosystemes():
    from matplotlib.transforms import offset_copy
    data = pickle.load(open('../analysis/ecosystemes.pickle', 'rb'))
    fig = graphs.scenarios(data)

    fig.axes[0].text(0, 1, "Lac St-Pierre", fontsize=34, weight='bold', color='#2b2929', alpha=.8, transform=offset_copy(fig.axes[0].transAxes, x=0, y=30, units='dots'))
    fig.savefig('../figs/Lac St-Pierre.png')

#
def ff_xls(site, save=False):
    from pandas.io.excel import ExcelWriter

    CP = analysis.CP
    i = CP['sites'].index(site)
    #W = ExcelWriter(fn)

    # Corresponding flow
    if site in ['var', 'pcl']:
        fs = ('stl',)
    elif site == 'srl':
        fs = ('stl', 'dpmi')


    out = {}
    for scen in ['bc', 'wd']:
        # Get level
        ts = FFio.level_series_QH(site, scen)

        # Sum flow contributions
        f = FFio.FF_flow(fs[0], scen)
        for s in fs[1:]:
            f = f.add( FFio.FF_flow(s, scen) )

        F = ts.valid().to_frame('Level m')
        F['Flow m3s'] = f
        y, q = F.index.levels
        F.index.set_levels((y+analysis.offset[scen], q), True)
        F = F.swaplevel(0,1)

        if save:
            P = F.to_panel()
            fn = 'Water_Levels_IJC_{0}_{1}_IGLD85.xlsx'.format(CP['names'][i], scen.upper())
            P.to_excel(fn)

        out[scen] = F
    if not save:
        return out


def chee():
    for site in ['srl', 'var']:
        ff_xls(site)
