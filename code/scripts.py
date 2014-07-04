import graphs, GLSLio, analysis
from matplotlib import pyplot as plt
import os
from imp import reload
reload(graphs)


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
def scenario_LSP():
    """Create Excel file storing LSP water levels for the base case and warm &
    dry scenarios."""
    from pandas.io.excel import ExcelWriter


    # Stations H Lac St-Pierre
    s = '02OC016'
    # Aussi '02OC003' Lac Sainte-Pierre a Louiseville (1961-1984) mais presque pas de données

    # Observations (1978-2010)
    h=GLSLio.get_hydat(s, 'H')
    M = GLSLio.get_station_meta(s)
    lat, lon = M['latitude'], M['longitude']

    # Datum ID: 431, Great Lakes International Datum 1985
    # From EC datum reference:
    # Station	Nom de la station	RN de référence	Altitude(ZC)	Altitude(SRIGL85)	SRIGL85(ZC)
    # 15975      Lac Saint-Pierre	 87L9000	   	 6.537	       9.980        	    3.443

    # Les données du modèles sont en Niveau Moyen des Mers, ce qui je pense,
    # correspond à l'altitude. Si on veut reporter ces valeurs au datum de la
    # station de mesure, on doit donc soustraire 6.537 des scénarios.




#
def save_ff_xls(site):
    from pandas.io.excel import ExcelWriter

    CP = analysis.CP
    i = CP['sites'].index(site)
    #W = ExcelWriter(fn)

    # Corresponding flow
    if site == 'var':
        fs = ('stl',)
    elif site == 'srl':
        fs = ('stl', 'dpmi')

    for scen in ['bc', 'wd']:
        # Get level
        ts = analysis.get_levels_ff(site, scen)

        # Sum flow contributions
        f = GLSLio.FF_flow(fs[0], scen)
        for s in fs[1:]:
            f = f.add( GLSLio.FF_flow(s, scen) )

        F = ts.valid().to_frame('Level m')
        F['Flow m3s'] = f
        y, q = F.index.levels
        F.index.set_levels((y+analysis.offset[scen], q), True)
        F = F.swaplevel(0,1)
        P = F.to_panel()

        fn = 'Water_Levels_IJC_{0}_{1}_IGLD85.xlsx'.format(CP['names'][i], scen.upper())
        P.to_excel(fn)

    #W.save()
