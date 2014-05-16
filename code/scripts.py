import graphs, GLSLio
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
