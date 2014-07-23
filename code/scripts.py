import graphs, GLSLio, FFio, analysis, GLSLutils, HYDATio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os, pickle
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

        # Compute flow at Trois-Rivières
        tsLSP = analysis.weight_EC_levels(analysis.EC_scen_Q['Trois-Rivieres'], wi)

        # Compute level at Lac St-Pierre
        lLSP = analysis.weight_EC_levels(ECL, wi)

        fn = 'Water_Levels_WhatIf2_Lac_St-Pierre_{0}_IGLD85.xlsx'.format(scen)
        fr = dump_xls(pd.Series(tsLSP, ts.index), pd.Series(lLSP, ts.index), os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))
        out['WI2'][scen] = fr

    pickle.dump(out, open('../analysis/ecosystemes.pickle', 'wb'))
    return out



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
