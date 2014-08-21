import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import basemap
from matplotlib.ticker import Formatter
import pandas as pd
import datetime as dt
import GLSLutils as util
import pickle

import GLSLio, FFio, ECio, HYDATio, GLSLutils
import analysis
from imp import reload
reload(GLSLio)
reload(analysis)
reload(util)

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Droid Sans'
c = '#00425C'
CC = dict(bc='Référence', ww='Chaud et humide', wd='Chaud et sec')


# see tricontour_smooth_delaunay.py

# Attention: les trois domaines n'ont pas l'air d'être sur le même datum (ça parait à la jonction du Lac St-Pierre)

# Haxby color palette
ncolors=11;
_haxby=np.array([[ 37, 57, 175], [40, 127, 251], [50, 190, 255], [106, 235, 255], [138, 236, 174], [205, 255, 162], [240, 236, 121], [255, 189, 87], \
    [255, 161, 68], [255, 186, 133], [255, 255, 255]])/255.;
haxby = mpl.colors.LinearSegmentedColormap.from_list('haxby', _haxby[::-1])


# Scenario colors
cs1 = '#124776'
cs2 = '#8d0d20'
cbc = '#333232'
cobs= '#2b2929'

class QOMFormatter(Formatter):
  def __init__(self, seq):
    self.seq = seq
  def __call__(self, x, pos=None):
    if pos is None or pos >= len(self.seq):
      return ''
    else:
      try:
        print (x, self.seq.index[x])
        return str(self.seq.index[x][0])
      except:
        return ''

def strip(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_position(('outward',10))
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()


def example_interpolation_debits():
    Q = analysis.EC_scen_Q
    x = Q['Sorel']
    y = Q['LaSalle']
    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(left=.15, bottom=.15, right=.9)

    ax.plot(x, y, '-', color=cbc, marker='o',  mfc='w', mec=cs1, lw=1.5, mew=1.5, label="Débits des huits scénarios du tableau 3")
    ax.set_xlabel("Débits à Sorel [m³/s]")
    ax.set_ylabel("Débits à LaSalle [m³/s]")
    ax.tick_params(labelright=True)

    xi = 8700
    yi = np.interp(xi, x,y)
    ax.plot([xi, xi], [ax.get_ylim()[0], yi], '--', color='gray', lw=1.5)
    ax.plot(ax.get_xlim(), [yi, yi], '--', color='gray', lw=1.5)

    ax.annotate("Débit reconstitué\nà Sorel", \
        (xi, ax.get_ylim()[0]), (.05,.55), textcoords='axes fraction', \
        arrowprops=dict(arrowstyle="->",  connectionstyle="arc3,rad=.4", color='gray'))

    ax.annotate("Débit correspondant à LaSalle", \
        (ax.get_xlim()[1], yi), (.45,.38), textcoords='axes fraction', \
        arrowprops=dict(arrowstyle="->",  connectionstyle="arc3,rad=-.2", color='gray'))

    ax.legend(fontsize='small', frameon=False, loc='upper left', numpoints=1)
    plt.savefig('../figs/exemple_interpolation_debits.png')
    return fig

def example_interpolation_niveaux():
    "LaSalle"
    Q = analysis.EC_scen_Q
    x = Q['Sorel']
    y = np.array([ 19.8039822 ,  20.2219321 ,  20.68718862,  21.12620549,
        21.70513717,  22.12565828,  22.73689272,  23.16902584])

    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(left=.15, bottom=.15, right=.9)

    ax.plot(x, y, '-', color=cbc, marker='o',  mfc='w', mec=cs1, lw=1.5, mew=1.5, label="Niveaux des huit scénarios stationnaires\ninterpolés à Pointe-Claire")
    ax.set_xlabel("Débits à Sorel [m³/s]")
    ax.set_ylabel("Niveaux à la station de Pointe-Claire [m]")

    ax.tick_params(labelright=True)

    xi = 8700
    yi = np.interp(xi, x,y)
    ax.plot([xi, xi], [ax.get_ylim()[0], yi], '--', color='gray', lw=1.5)
    ax.plot(ax.get_xlim(), [yi, yi], '--', color='gray', lw=1.5, )

    ax.annotate("Débit reconstitué\nà Sorel", \
        (xi, ax.get_ylim()[0]), (.05,.55), textcoords='axes fraction', \
        arrowprops=dict(arrowstyle="->",  connectionstyle="arc3,rad=.4", color='gray'))

    ax.annotate("Niveau correspondant\nà Pointe-Claire", \
        (ax.get_xlim()[1], yi), (.55,.4), textcoords='axes fraction', \
        arrowprops=dict(arrowstyle="->",  connectionstyle="arc3,rad=-.2", color='gray'))

    plt.savefig('../figs/exemple_interpolation_niveaux.png')

    ax.legend(fontsize='small', frameon=False, loc='upper left', numpoints=1)
    plt.savefig('../figs/exemple_interpolation_niveaux.png')
    return fig

def showcase_scenarios(var='H'):

    keys = {'Q':'Flow m3s', 'H':'Level m'}

    Fbc, F1, F2 = analysis.scenarios_Sorel()


    fig, ax = plt.subplots(nrows=1, figsize=(10,5))
    fig.subplots_adjust(left=.06, right=.98)

    axt = plt.twiny(ax)

    a = .6
    i = 0
    name = keys[var]

    Lbc = axt.plot_date(util.ordinal_qom(Fbc[name]), Fbc[name].values, '-', color=cbc, lw=.6, alpha=a,  label='Référence (1953–2012)')[0]
    Ls1 = ax.plot_date(util.ordinal_qom(F1[name]), F1[name].values, '-', color=cs1, lw=1., alpha=a, label='What-if #1 (2040–2069)')[0]
    Ls2 = ax.plot_date(util.ordinal_qom(F2[name]), F2[name].values, '-', color=cs2, lw=1., alpha=a, label='What-if #2 (2010–2069)')[0]
    ax.legend((Lbc, Ls1, Ls2), (Lbc.get_label(), Ls1.get_label(), Ls2.get_label()), loc='upper right', frameon=False, ncol=3, fontsize='small')

    if var == 'H':
        ax.set_ylabel('Niveau à Sorel [m]')
    elif var == 'Q':
        ax.set_ylabel('Débit à Sorel [m³/s]')


    if 0:
        L = HYDATio.get_hydat('02OJ022', 'H') #H
        LQ = GLSLutils.group_qom(L).mean()
        LQ.index.names = ["Year", "QTM"]
        axest[1].plot_date(util.ordinal_qom(LQ), LQ.values, '-', color='green', lw=1.)

    if 1:
        ax.set_xlim(dt.datetime(2050,1,1), dt.datetime(2065,1,1))
        axt.set_xlim(dt.datetime(1993,1,1), dt.datetime(2008,1,1))

    plt.setp(ax.get_xticklabels(), color=cs2)
    plt.setp(axt.get_xticklabels(), color=cbc)

    fig.savefig('../figs/resume_scenarios_zoom.png')



def scenarios(data):

    fig, axes = plt.subplots(3, 1, figsize=(14,11), sharey=True)
    fig.subplots_adjust(left=.06, right=.98, bottom=.04)
    lw = .5

    axes[0].xaxis.tick_top()
    axes[0].plot_date(util.ordinal_qom(data['obs']), data['obs'].values, '-', lw=lw, color=cobs, label='Observations')
    axes[0].legend(frameon=False, loc='upper right')


    tax1 = plt.twiny(axes[1])
    bc1 = data['WI1']['BC']['Level m']
    sc1 = data['WI1']['WD']['Level m']
    L11 = tax1.plot_date(util.ordinal_qom(bc1), bc1.values, '-', color=cbc, lw=lw, label="Base case #1")[0]
    L12 = axes[1].plot_date(util.ordinal_qom(sc1), sc1.values, '-', color=cs1, lw=lw, label="Scenario #1")[0]
    axes[1].legend((L11, L12), ("Scénario de référence #1", "Scénario futur #1 - Chaud et sec"), frameon=False, loc="upper right")

    tax2 = plt.twiny(axes[2])
    bc2 = data['WI2']['BC']['Level m']
    sc2 = data['WI2']['SC']['Level m']
    L21 = tax2.plot_date(util.ordinal_qom(bc2), bc2.values, '-', color=cbc, lw=lw, label="Base case #2")[0]
    L22 = axes[2].plot_date(util.ordinal_qom(sc2), sc2.values, '-', color=cs2, lw=lw, label="Scenario #2")[0]
    axes[2].legend((L21, L22), ("Scénario de référence #2", "Scénario futur #2 - Amplication saisonnière"), frameon=False, loc="upper right")

    plt.setp(axes, ylabel="Niveau d'eau [m]")

    plt.setp(axes[0].get_xticklabels(), color=cobs)
    plt.setp(axes[1].get_xticklabels(), color=cs1)
    plt.setp(axes[2].get_xticklabels(), color=cs2)
    plt.setp(tax1.get_xticklabels(), color=cbc)
    plt.setp(tax2.get_xticklabels(), color=cbc)

    return fig




def graph_flow_level(flow, level):
    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(flow, level, 'k.', alpha=.6)
    ax.set_xlabel("Débit [m³/s]")
    ax.set_ylabel("Niveau [m]")

    return fig


def NBS_qm_correction(n=10):
    qm = analysis.QM_NBS(n)


    cm = plt.cm.jet(np.linspace(0,1,4))
    fig, axes = plt.subplots(nrows=4, figsize=(8.5, 11))
    gcms = ['CGCM3.1', 'ECHAM5', 'CNRM-CM3', 'CGCM2.3']

    for i, s in enumerate(['winter', 'spring', 'summer', 'fall']):
        axes[i].set_color_cycle(cm)
        axes[i].text(.05, .9, s.capitalize(), transform=axes[i].transAxes)

        for gcm, vals in qm[s].items():
            j = gcms.index(gcm)
            #axes[i].plot(range(0,100,10), vals, label=gcm)
            axes[i].bar(np.linspace(0, 100, n)+j*2, vals, 1.9, color=cm[j], label=gcm)

    axes[0].legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.05))
    axes[-1].set_xlabel('Percentile')
    plt.setp(axes, ylabel='ΔNBS')

def NBS_cycle_GL(stat=np.mean):
    """Plot the annual NBS cycle."""
    import ndict

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    res = ndict.NestedDict()
    for m in months:
        res[m] = GLSLio.NBS(m, stat)
        wa = GLSLio.basins_weighted_NBS_average(res[m])
        res[m]['lacgreat lakes'] = wa
        res[m].pop('lacMHG')

    nbs = ndict.NestedDict()
    for l in res.keylevel(1):
        for a in res.keylevel(2):
            nbs[l][a] = [res[m][l][a] for m in months]


    fig, axes = plt.subplots(2, figsize=(8,6))
    fig.subplots_adjust(right=.97)

    lake = 'lacgreat lakes'; i=0
    ax = axes[0]
    ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
    for (r,f) in analysis.aliases.items():
        ax.plot(nbs[lake][r], marker='o', lw=1, mec='none', label=f)
    ax.plot(nbs[lake]['obs'], ms=10, lw=2, color='#272727', label='Obs.'    )
    ax.set_ylabel('NBS (Référence) mm/j')

    ax = axes[1]
    ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
    for (r,f) in analysis.aliases.items():
        ax.plot(np.array(nbs[lake][f]) - np.array(nbs[lake][r]), marker='o', lw=1, mec='none', label='{0}/{1}'.format(r,f))
    ax.axhline(0, color='gray', zorder=-1)
    #ax.text(.02, .7, 'Great Lakes CC', ha='left', va='baseline', size='large', weight='bold', color='#272727', transform=ax.transAxes)
    ax.set_xlabel('Mois')
    ax.set_ylabel('ΔNBS')


    ax.legend(loc='upper right', fontsize='small', frameon=False, numpoints=1, ncol=5)

    fig.savefig('../figs/NBS_annual_cycle_full_GL.png')

def NBS_cycle_full(stat=np.mean):
    """Plot the annual NBS cycle."""
    import ndict

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    res = ndict.NestedDict()
    for m in months:
        res[m] = GLSLio.NBS(m, stat)
        wa = GLSLio.basins_weighted_NBS_average(res[m])
        res[m]['lacgreat lakes'] = wa
        res[m].pop('lacMHG')

    nbs = ndict.NestedDict()
    for l in res.keylevel(1):
        for a in res.keylevel(2):
            nbs[l][a] = [res[m][l][a] for m in months]

    fig = plt.figure(figsize=(8.5,11))
    gs = plt.GridSpec(8,1)
    loc = dict(lacontario=(0,0), lacerie=(1,0), lachuron=(2,0), lacmichigan=(3,0), lacsuperior=(4,0), )
    loc['lacgreat lakes'] = (slice(5,7), 0)

    lakes = nbs.keylevel(0)
    lnames = [s[3:].title() for s in lakes]

    for i, lake in enumerate(lakes):
        ax = fig.add_subplot(gs[loc[lake]])
        ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
        for (r,f) in analysis.aliases.items():
            ax.plot(nbs[lake][r], marker='o', lw=1, mec='none', label=r)
        ax.plot(nbs[lake]['obs'], ms=10, lw=2, color='#272727', label='Obs.'    )

        ax.text(.02, .7, lnames[i], ha='left', va='baseline', size='large', weight='bold', color='#272727', transform=ax.transAxes)

    ax = fig.add_subplot(gs[7,0])
    ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
    for (r,f) in analysis.aliases.items():
        ax.plot(np.array(nbs[lake][f]) - np.array(nbs[lake][r]), marker='o', lw=1, mec='none', label='{0}/{1}'.format(r,f))
    ax.axhline(0, color='gray', zorder=-1)
    ax.text(.02, .7, 'Great Lakes CC', ha='left', va='baseline', size='large', weight='bold', color='#272727', transform=ax.transAxes)

    ax = fig.axes[-3]
    ax.set_ylabel('NBS (Référence) mm/j')

    ax.set_xlabel('Mois')
    ax.legend(loc='lower right', fontsize='small', frameon=False, numpoints=1)

    fig.savefig('../figs/NBS_annual_cycle_full.png')

def scenario_2():
    bc, s2 = analysis.scenario_2()

    fig, ax = plt.subplots(figsize=(10,4.5))
    fig.subplots_adjust(left=.1, right=.98)
    axt = plt.twiny()


    Lbc = axt.plot_date(util.ordinal_qom(bc), bc.values, '-', color=cbc, lw=.6, alpha=.9,  label='Débits reconstitués à Sorel (1953–2012)')[0]

    # Scenario 2
    Ls2 = ax.plot_date( util.ordinal_qom(s2), s2.values, '-', color=cs2, lw=1., label='Débits scénario #2 (2010–2069)')[0]

    ax.set_ylabel('Débit à Sorel [m³/s]')

    ax.legend((Lbc, Ls2), (Lbc.get_label(), Ls2.get_label()), loc='upper right', frameon=False)

    ax.set_xlim(dt.datetime(2009,6,1).toordinal(), dt.datetime(2070,6,1).toordinal())
    axt.set_xlim(dt.datetime(1952,6,1).toordinal(), dt.datetime(2013,6,1).toordinal())
    plt.setp(ax.get_xticklabels(), color=cs2)
    plt.setp(axt.get_xticklabels(), color=cbc)

    fig.savefig('../figs/scenario2.png')


def explain_scenario_2():
    r, f = analysis.NBS_delta()
    bc, s2 = analysis.scenario_2()

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    fig, axes = plt.subplots(nrows=2)
    fig.subplots_adjust(top=.87, hspace=.3, right=.89, left=.1)

    # Sorel
    y = bc.groupby(level=1).mean()
    x = np.linspace(0,11, 48)

    axes[0].plot(r, label="Période de référence", color='b', lw=2)
    axes[0].plot(f, label="Période future", color='orange', lw=2)
    axes[0].set_ylabel('NBS [mm/j]')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0,1), frameon=False, fontsize='small', title="Cycle NBS de la simulation afa/afd")
    ax = plt.twinx(axes[0])
    ax.plot(x, y, color='k', lw=2, label="Débit à Sorel [m³/s]")
    ax.set_ylabel('Débit à Sorel [m³/s]')
    ax.legend(loc='lower right', bbox_to_anchor=(1,1), frameon=False, fontsize='small')

    axes[1].plot(f-r, label="Δ NBS", color='k', lw=2)
    axes[1].axhline(0, color='gray')
    #cc = np.roll(np.interp(x, range(12), (f-r)/r.ptp()) * y.ptp(), 2)
    cc = s2.groupby(level=1).mean()
    ax2 = plt.twinx(axes[1])
    ax2.plot(x, cc-y, 'r-', lw=2, label='Δ Débit à Sorel')


    plt.setp(axes, xticks=range(12), xlim=(0,11))
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([m.capitalize() for m in months])
    axes[1].legend(loc='lower left', bbox_to_anchor=(0,1), frameon=False, fontsize='small')
    ax2.legend(loc='lower right', bbox_to_anchor=(1,1), frameon=False, fontsize='small')
    axes[1].set_ylabel('Δ NBS [mm/j]')
    ax2.set_ylabel('Δ Débit [m³/s]')
    axes[1].set_xlabel("Mois")

    fig.savefig('../figs/explanation_scenario_2.png')
    return fig
    #
def NBS_cycle_model_average():

    ref = {}; fut = {}
    for m in months:
        nbs = GLSLio.NBS(m)
        wa = GLSLio.basins_weighted_NBS_average(nbs)
        ref[m] = np.mean([wa[a] for a in analysis.aliases.keys()])
        fut[m] = np.mean([wa[a] for a in analysis.aliases.values()])

    fig, axes = plt.subplots(nrows=2)
    fig.subplots_adjust(top=.87, hspace=.3, right=.89, left=.1)
    r = np.array([ref[m] for m in months])
    f = np.array([fut[m] for m in months])

    # Sorel
    ts = GLSLio.Q_Sorel('qtm')
    y = ts.groupby(level=1).mean()
    x = np.linspace(0,11, 48)

    axes[0].plot(r, label="Reference annual NBS cycle", color='b', lw=2)
    axes[0].plot(f, label="Future annual NBS cycle", color='orange', lw=2)
    axes[0].set_ylabel('NBS [mm/j]')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0,1), frameon=False, fontsize='small')
    ax = plt.twinx(axes[0])
    ax.plot(x, y, color='k', lw=2, label="Débit à Sorel [m³/s]")
    ax.set_ylabel('Sorel flow [m³/s]')
    ax.legend(loc='lower right', bbox_to_anchor=(1,1), frameon=False, fontsize='small')

    axes[1].plot(f-r, label="CC NBS", color='k', lw=2)
    axes[1].axhline(0, color='gray')
    cc = np.roll(np.interp(x, range(12), (f-r)/r.ptp()) * y.ptp(), 2)
    ax2 = plt.twinx(axes[1])
    ax2.plot(x, cc, 'r-', lw=2, label='CC Sorel flow [m³/s]')


    plt.setp(axes, xticks=range(12), xlim=(0,11))
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([m.capitalize() for m in months])
    axes[1].legend(loc='lower left', bbox_to_anchor=(0,1), frameon=False, fontsize='small')
    ax2.legend(loc='lower right', bbox_to_anchor=(1,1), frameon=False, fontsize='small')
    axes[1].set_ylabel('Δ NBS [mm/j]')
    ax2.set_ylabel('Δ Débit [m³/s]')
    axes[1].set_xlabel("Mois")

    fig.savefig('../figs/NBS_annual_cycle.png')
    return fig
#
def NBS_scatter(freq='annual', stat=np.mean):
    """Plot the future vs reference NBS for each lake.
    """
    nbs, aliases = GLSLio.NBS(freq, stat)

    # All GL
    wa = GLSLio.basins_weighted_NBS_average(nbs)
    nbs['lacgreat lakes'] = wa

    nbs.pop('lacMHG')
    lakes = nbs.keylevel(0)
    lnames = [s[3:].title() for s in lakes]

    #fig, axes = plt.subplots(ncols=5, figsize=(14,8.5))
    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(3,3)
    loc = dict(lacontario=(0,0), lacerie=(1,0), lachuron=(2,0), lacmichigan=(0,1), lacsuperior=(0,2), )
    loc['lacgreat lakes'] = (slice(1,None), slice(1,None))

    for i, lake in enumerate(lakes):

        ax = fig.add_subplot(gs[loc[lake]])
        ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
        for (r,f) in aliases.items():
            ax.plot([nbs[lake][r],], [nbs[lake][f],], marker='o', lw=0, mec='none', label='{0}/{1}'.format(r,f))
        ax.plot([nbs[lake]['obs'],], [nbs[lake]['obs'],], marker='+', ms=10, lw=0, mec='#272727', mew=2, label='Obs.'    )

        ax.text(.05, .8, lnames[i], ha='left', va='baseline', size='large', weight='bold', color='#272727', transform=ax.transAxes)
        ax.set_aspect('equal', adjustable='datalim')

    ax = fig.axes[-2]
    ax.set_xlabel('NBS (Référence) mm/j')
    fig.axes[2].set_ylabel('NBS (Futur) mm/j')
    ax.legend(loc='lower right', fontsize='small', frameon=False, numpoints=1)

    for ax in fig.axes:
        ax.set_autoscale_on(False)
        ax.plot([-10,10], [-10,10], color='grey', alpha=.5, lw=.5)
#
def plot_FF_flow(site):
    """Plot time series of the flow.
    """
    from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
    # Plot only the first 29 years
    # The next years are not a perfect copy of the first years, I suspect because
    # of the storage effect.
    #if scen == 'bc':
    yo = 1961
    c = dict(bc='#393737', wd='#ab4918', ww='#106b86')

    fig, ax = plt.subplots(1,1, figsize=(8,4), facecolor='w')

    fig.subplots_adjust(right=.97, top=.95)
    strip(ax)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.autoscale(enable=True, tight=True)
    ax.set_ymargin(.05)

    # QOM series
    for scen in ['bc', 'ww', 'wd']:
        ts = FFio.FF_flow(site, scen, yo)[:29*48]

        # Annual min
        g = ts.groupby(level=0)
        tm = g.min()
        itm = g.idxmin()
        tm_d = [d.toordinal() for d in util.qom2date(itm)]


        if scen == 'bc':
            ax.plot_date(util.ordinal_qom(ts), ts.values, '-', color='#393737', alpha=.5)

        ax.plot_date(tm_d, tm.values, '-', color=c[scen], marker='o',  mfc='w', mec=c[scen], label=CC[scen], lw=1.5, mew=1.5)

    leg = ax.legend(loc='upper left', frameon=False, numpoints=1, title='Minimum annuel', fontsize='small')
    ax.set_ylabel("Débit [m³/s]")

    return fig, ax



def plot_station_qom_levels(sid):
  """Graphic of the mean QOM level time series."""
  ts = GLSLio.get_hydat(sid, 'H')

  # Average at the QOM scale
  gr = GLSLio.group_qom(ts)
  ts = gr.mean()


  fig, ax = plt.subplots(1, 1, figsize=(10, 4))
  fig.subplots_adjust(bottom=.1, right=.95, left=.1)
  ts.plot(ax=ax, color=c, lw=1.5)
  ts.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)

  ax.grid(ls='-', lw=.1, color='#777777')
  ax.grid(axis='x')

  ax.set_ylabel("Niveau [m]")


  ax.set_xlim(0, len(ts))

  years = ts.index.levels[0]
  if len(years) > 20:
    xt = []; t = []
    y1, y2 = np.around(years[0]+4,-1), np.around(years[-1]-4,-1)
    for i, y in enumerate(np.arange(y1, y2+1, 10)):
       xt.append(ts.index.get_loc(y).start)
       t.append(str(y))

  ax.set_xticks(xt)
  ax.set_xticklabels(t)

  return fig, ax


def plot_station_qom_level_stats(sid):
    """Graphic of the statistics of the level time series."""
    import itertools

    dts = GLSLio.get_hydat(sid, 'H').resample('D')

    # Average at the QOM scale
    gr = GLSLio.group_qom(dts)
    n = gr.count()
    qs = pd.Series(data=np.where(n >= 4, gr.mean(), np.nan), index=n.index)

    g = qs.groupby(level=0)
    m = [g.count() < 30]
    set_mask = lambda x: np.where(g.count() > 30, x, np.nan)

    ts = GLSLio.qom2ts(qs)

    years = g.mean().index
    i = [dt.datetime(y, 6, 1) for y in years]
    fb = zip(years, itertools.cycle([1,]))

    tsm = pd.TimeSeries(data=set_mask(g.mean().values), index=i)
    tsmin = pd.TimeSeries(data=set_mask(g.min().values), index=GLSLio.qom2date(g.idxmin()))
    tsmax = pd.TimeSeries(data=set_mask(g.max().values), index=GLSLio.qom2date(g.idxmax()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='w')
    fig.subplots_adjust(bottom=.1, right=.97, left=.07, top=.9)
    strip(ax)


    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.autoscale(enable=True, tight=True)
    ax.set_ymargin(.05)

    ax.plot_date(ordinal(ts), ts.values, '-', lw=.5, color='#393737', alpha=.5)
    ax.plot_date(ordinal(tsm), tsm.values, '-', marker='o', color='#393737', mfc='w', mec='#393737', mew=1.5, lw=1.5, label='Moyenne')
    ax.plot_date(ordinal(tsmin), tsmin.values, '-', marker='o', color='#810202', mfc='w', mec='#810202', mew=1.5, lw=1.5, label='Minimum')
    ax.plot_date(ordinal(tsmax), tsmax.values, '-', marker='o', color='#3866b7', mfc='w', mec='#3866b7', mew=1.5, lw=1.5, label='Maximum')


    #ax.fill_between(tsm.index, tsmax, tsmin, color="#333333", alpha=.1)

    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')

    ax.set_ylabel("Niveau [m]")
    ax.legend(loc='lower right', ncol=3, frameon=False, numpoints=1, bbox_to_anchor=(1,.95), fontsize='small', title='Statistiques annuelles')
    return fig, ax



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

def Sorel_annual_cycle():
    fig, ax = plt.subplots(1)
    ts = GLSLio.Q_Sorel('qtm')
    y = ts.groupby(level=1).mean()
    y.plot(ax=ax)
    ax = plt.gca()
    ax.set_ylabel('Débit [m³/s]')
    ax.set_xlim(0,48)


def plot_Sorel_annual_minimum_qom_levels_flows():


    # Load levels
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15930-1-JAN-1916_slev.csv')
    level_min = GLSLio.annual_min_qom_ts(ra)

    # Load streamflow
    ts = GLSLio.Q_Sorel('qtm')
    g = ts.groupby(level=0)
    n = g.count()
    q_min = pd.TimeSeries(np.where(n>30, g.min(), np.nan), n.index)

    # Graph
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, facecolor='w')
    fig.subplots_adjust(bottom=.07, right=.95, left=.1, hspace=.1, top=.92)
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
        strip(ax)
        ax.set_ymargin(.07)
        ax.autoscale(enable=True, axis='y', tight=True)


    axes[1].set_xlim(right=2015)
    plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)

    for y in [2001, 2007, 2012]:
        axes[1].annotate(str(y), (y, level_min[y]), (y, .9),
            ha='center', size=11,
            textcoords=trans[1],
            arrowprops=dict(arrowstyle='-', fc='w', ec='k', shrinkB=5,), color='k')

    axes[1].text(1932, 0, 'Zéro des cartes', size=10, ha='left', )

    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="k", ec="k", lw=2)
    t = axes[0].text(1961, .01, "Régularisation des débits  ", ha="left", va="bottom",
            size=12, weight='bold', color='white', transform=trans[0],
            bbox=bbox_props)

    return fig, axes
    #plt.savefig('../figs/Sorel_q_l_min.png')
    #plt.savefig('../figs/Sorel_q_l_min.svg')


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



def plot_mesh():
    import matplotlib.transforms as mtransforms
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    cm = mpl.cm.spectral

    fig, ax = plt.subplots(1,1, figsize=(8,4))
    fig.set_facecolor('w')
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)


    AR = mtransforms.Affine2D()
    AR.rotate_deg(-25)
    pts = np.hstack(GLSLio.EC_pts().values())
    X, Y, Z = pts
    S = ax.scatter(X, Y, c=Z, s=.3, linewidths=0, transform=AR + ax.transData, cmap=cm)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.set_xlim(2355000, 2515000)
    ax.set_ylim(4420000, 4500000)
    ax.set_axis_off()

    axins = zoomed_inset_axes(ax, 10, loc=4) # zoom =
    axins.scatter(X, Y, c=Z, s=1, linewidths=0, transform=AR + axins.transData, cmap=cm)

    # sub region of the original image
    x1, y1 = 2459000, 4480000
    axins.set_xlim(x1, x1+5E3)
    axins.set_ylim(y1, y1+5E3)

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    axins.set_xticks([]); axins.set_yticks([])

    cbax = fig.add_axes([.1,.9,.45,.04])
    cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
    cb.set_label('Élévation [m]')

    data = []
    def onClick(event):
        data.append((event.xdata, event.ydata))

    def close(evt):
        print(data)
        return data

    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('close_event', close)

    return fig, ax


def plot_depth_map(scen, pts={}, inset=True):
    from scipy.spatial import Delaunay
    import matplotlib.transforms as mtransforms
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset


    cm = haxby
    #cm.set_over('w')


    #m = basemap.Basemap(projection='tmerc', resolution='c', lat_0=45.8, lon_0=-73.5, k_0=0.99990, ellps='GRS80', width=170000, height=170000)
    proj = ECio.MTM8()
    #m.drawcoastlines()
    #m.drawcountries()
    #m.drawrivers()

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    cbax = fig.add_axes([.1,.9,.45,.04])
    fig.set_facecolor('w')
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)

    AR = mtransforms.Affine2D()
    AR.rotate_deg(-25)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.set_xlim(2355000, 2515000)
    ax.set_ylim(4420000, 4500000)
    ax.set_axis_off()

    # Plot control points
    cp = False
    if cp:
        for key, c in zip(analysis.CP['sites'], analysis.CP['coords']):
            x,y = proj(*c)
            ax.plot([x,], [y,], 'o', mfc='w', mec='b', ms=10, label=key, zorder=1, transform=AR + ax.transData)

        plt.legend()

    if pts:
        for key, c in pts.items():
            x,y = proj(*c)
            ax.plot([x,], [y,], '*', mfc='k', mec='none', ms=4, label=key, zorder=1, transform=AR + ax.transData)

        #ax.legend(fontsize='small')


    pts = np.hstack(ECio.EC_pts().values())
    D = np.hstack(ECio.EC_depth(scen=scen).values())

    X, Y, Z = pts
    D = np.ma.masked_less_equal(D, 0)
    S = ax.scatter(X, Y, c=D, s=1, linewidths=0, transform=AR + ax.transData, cmap=cm, vmax=45, vmin=1e-2, norm=mpl.colors.LogNorm())



        # Convert coordinates into lat, long
        #lon, lat = proj(x, y, inverse=True)

        # Convert back into map projection
        #x, y  = m(lon, lat)

        # Construct Delaunay tesselation
        # I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
        #T = analysis.get_tesselation(reg)

        #plt.tricontourf(T, d, 20, mask=ma, vmin=0, vmax=20, cmap=cm)
        #plt.tripcolor(T, d, vmin=0, vmax=20, cmap=cm)

    cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
    cb.set_label('Profondeur [m]')
    #cb.formatter(ScalarFormatter(cb.ticks)
    if inset:
        axins = zoomed_inset_axes(ax, 3.5, loc=4, borderpad=0.3) # zoom =
        axins.scatter(X, Y, c=D, s=2, cmap=cm, vmax=45, vmin=1e-2, linewidths=0, transform=AR + axins.transData, norm=mpl.colors.LogNorm())

        # sub region of the original image
        x1, y1 = 2475000, 4482000
        axins.set_xlim(x1, x1+2.2E4)
        axins.set_ylim(y1, y1+1.1E4)

        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.7")
        axins.set_xticks([]); axins.set_yticks([])

    return fig, ax


def plot_map(bg=False):
    """Draw a nice map of the study region with the CCSM3 grid lines
    and the contour of a set of regions.

    Parameters
    ----------
    bg : bool
      Draw a nice color background
    regions : dict
      A dictionary of Region objects to draw.
    ccsm3 : bool
      Draw the CCSM3 atmospheric model grid.
    """


    # Create the projection and draw the state boundaries.
    fig = plt.figure(figsize=(8,4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111)
    map = basemap.Basemap(projection='stere',
                  lat_0=46., lon_0=-73.5, lat_ts=45, width=.6e6, height=.3e6,
                  resolution='i')
    map.drawstates(color='gray')
    map.drawcountries()
    #map.bluemarble()


    # Plot control points
    cp = True
    if cp:
        for name, c in zip(analysis.CP['names'], analysis.CP['coords']):
            x,y = map(*c)
            M = ax.plot([x,], [y,], 'o', mfc='w', mec='orange', ms=6, label=name, zorder=1, mew=2)
            ax.annotate(name, (x,y), (10, -3), textcoords='offset points')

    if bg:
        image = '/home/david/data/earth_image/NE1_HR_LC_SR_W_DR.tif'
        geo = dict(lon0=-180, lat0=90, dx=.016666666666, dy=-0.01666666666, x_aff=0, y_aff=0)
        lon, lat, bgim = crop_tiff(image, map.lonmin, map.latmin, map.lonmax, map.latmax, geo) # (3, ny, nx)

        im = np.empty((len(lat), len(lon), 3), np.uint8)
        for k in range(3):
            im[:,:,k] = map.transform_scalar(bgim[:, :, k], lon, lat[::-1], len(lon), len(lat))
        map.imshow(im, origin='upper')
    else:
        map.drawcoastlines()

    map.drawmapscale(-71., 45.0, -73.5, 46, 100, barstyle='fancy')

    return map

def crop_tiff(gtif, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, geo={}):
    import PIL


    I = PIL.Image.open(gtif)
    # lon0, lat0 corresponds to the node coordinate.
    # In the twf file, what is stored is the tracer coordinate.

    nx,ny = I.size
    lon0 = geo['lon0']
    lat0 = geo['lat0']
    dx = geo['dx']
    dy = geo['dy']
    x_aff = geo['x_aff']
    y_aff = geo['y_aff']


    lon1 = lon0 + nx*dx    + ny*x_aff
    lat1 = lat0 + nx*y_aff + ny*dy

    imin = int((llcrnrlon - lon0) / dx)
    imax = int((urcrnrlon - lon0) / dx)+1

    jmax = int((llcrnrlat - lat0) / dy)+1
    jmin = int((urcrnrlat - lat0) / dy)

    if x_aff == 0 and y_aff==0:
        lon = lon0 + dx/2.+ np.arange(imin, imax) * dx
        lat = lat0 + dy/2. + np.arange(jmin, jmax) * dy

    A = np.asarray(I.crop([imin, jmin, imax, jmax]))

    return lon, lat, A
