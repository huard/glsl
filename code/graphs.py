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

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


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

oqs = GLSLutils.ordinal_qom

def oq(y,m):
	return GLSLutils.qom2date([(y,m)])[0].toordinal()

def strip(ax):
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['bottom'].set_position(('outward',10))
	ax.yaxis.tick_left()
	ax.xaxis.tick_bottom()

def make_patch_spines_invisible(ax):
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for key, sp in ax.spines.items():
		sp.set_visible(False)

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
	plt.savefig('../figs/exemple_interpolation_niveaux.png', dpi=300)
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



def zcmap(ts, vmin=-.5, ax=None, cb=True):
	from matplotlib import colors

	if ax is None:
		fig, ax = plt.subplots(1,1, figsize=(8,6), facecolor='w')

	p = ts.to_frame().to_panel()
	x = p.values[0].T * 100

	gh = plt.get_cmap('gist_heat')
	cm = colors.ListedColormap([gh(i) for i in np.linspace(.05, .8, 6)])
	cm.set_over('white')
	PM = ax.pcolormesh(p.major_axis.values, p.minor_axis.values, x, cmap=cm, vmin=vmin*100, vmax=10)

	ax.invert_yaxis()
	ax.autoscale(tight=True)

	ax.set_ylabel('Période')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')

	#ax.set_yticks(2.5+np.arange(0, 48, 12))
	#ax.set_yticklabels(['Hiver', 'Printemps', 'Été', 'Automne'])
	ax.set_yticks(range(0,48,4), )
	ax.set_yticks(np.arange(2,48,4), minor=True )
	ax.set_yticklabels([])
	ax.set_yticklabels([l[0].upper() for l in months], minor=True)
	ax.grid(axis='y', ls='-', alpha=.3, lw=.5)

	if cb:
		cb = plt.colorbar(PM, ax=ax, extend='max', shrink=.8)
		cb.set_ticks(np.linspace(-50, 10, 7))
		cb.set_label("Niveau [cm]")

	return ax


def souszero_figs(s, zc, name=''):
	meta = HYDATio.get_station_meta(s)
	lon, lat = meta['LONGITUDE'], meta['LATITUDE']
	EC = analysis.interpolate_EC_levels(lon, lat).tolist()
	scens = analysis.scenarios_H(lat=lat, lon=lon, EC=EC)
	fig, axes = plt.subplots(1,3, figsize=(14,6), facecolor='w')

	i = 0
	for ts, tag in zip(scens, ["REF", "WI1", "WI2"]):
		ax = zcmap(ts - zc, ax=axes[i], cb=i > 1)

		i += 1

	axes[0].text(.98, .9, "Occurrence de niveaux\nsous le 10 cm du \nzéro des cartes", transform=axes[0].transAxes, ha='right', va='top')
	axes[0].text(.98, .98, name, transform=axes[0].transAxes, size='x-large', color='#272727', alpha=.7, weight='bold', va='top', ha='right')
	fig.tight_layout()
	fig.savefig('../deliverables/Tourisme/Souszero_{0}.png'.format(name))
	plt.close()



def tourisme():
	"""Graphique des depassements du zero des cartes.

	Pas sur que mon équivalence station/ZC est bonne.
	"""

	args = [('02OA039', 20.351, 'Pointe-Claire'),
			('02OC016', 3.443, 'Lac St-Pierre'),
			('02OJ033', 3.8, 'Sorel')]

	pts = {}
	for arg in args:
		s, zc, name = arg
		meta = HYDATio.get_station_meta(s)
		pts[name] = meta['LONGITUDE'], meta['LATITUDE']
		souszero_figs(s, zc, name)

	fig, ax = plot_depth_map(4, pts, var='level')
	fig.savefig('../deliverables/Tourisme/Carte.png')
	plt.close()



def explain_extension(site):
	from mpl_toolkits.axes_grid1.inset_locator import mark_inset

	ref = FFio.total_flow(site, 'bc')
	fut = FFio.total_flow(site, 'wd')

	s1 = GLSLutils.select_and_shift(ref, before=1979, after=1988, offset=36)
	s2 = GLSLutils.select_and_shift(fut, before=2049, after=2063, offset=-24)

	s = analysis.extend_WI1(ref, fut)
	s = s.reindex(s.index.truncate(after=2065))

	fig, axes = plt.subplots(3, 1, figsize=(8,8))
	fig.subplots_adjust(bottom=.05, top=.97, right=.97)

	bc = ref.ix[:1990]
	wd = fut.ix[:2065]

	a=.6

	axes[0].plot_date(oqs(bc), bc.values, '-', color=cbc, lw=1.5)
	axes[0].plot_date(oqs(wd), wd.values, '-', color=cs1, lw=1.5)
	axes[0].text(.25, .8, "Scénario Base Case\nde la CMI (1962-1990)", transform=axes[0].transAxes)
	axes[0].text(.7, .77, "Scénario Chaud&Sec\nde la CMI (2040-2069)", transform=axes[0].transAxes)


	axes[1].plot_date(oqs(bc), bc.values, '-', color=cbc, lw=1.5)
	axes[1].plot_date(oqs(bc.ix[1979:1988]), bc.ix[1979:1988].values, '-', color='#868585', lw=1.5)
	axes[1].plot_date(oqs(wd), wd.values, '-', color=cs1, lw=1.5)
	axes[1].plot_date(oqs(wd.ix[2049:2063]), wd.ix[2049:2063].values, '-', color='#669ac5', lw=1.5)
	axes[1].plot_date(oqs(s1), s1.values, '-', color='#868585', lw=1.5, alpha=a)[0]
	axes[1].plot_date(oqs(s2), s2.values, '-', color='#669ac5', lw=1.5, alpha=a)

	axes[1].annotate("", [oq(2015, 1), s1.ix[2015].max()], [oq(1986, 1), bc.ix[1986].max()], \
		xycoords='data', textcoords='data',
		arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5",))

	axes[1].annotate("", [oq(2040, 1), s2.ix[2039].max()], [oq(2050, 1), wd.ix[2050].max()], \
		xycoords='data', textcoords='data',
		arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5",))
	axes[1].set_ylabel("Débit [m³/s]")
	axes[1].text(.12, .2, '1979-1988', transform=axes[1].transAxes)
	axes[1].text(.82, .15, '2049-2063', transform=axes[1].transAxes)

	axes[1].text(.46, .85, '2015-2024', transform=axes[1].transAxes)
	axes[1].text(.59, .78, '2025-2049', transform=axes[1].transAxes)

	sz = s.ix[2015:2039]
	axes[2].plot_date(oqs(s1), s1.values, '-', color='#868585', lw=1.5, alpha=a)[0]
	axes[2].plot_date(oqs(s2), s2.values, '-', color='#669ac5', lw=1.5, alpha=a)
	axes[2].plot_date(oqs(sz), sz.values, '-', color=cs1, lw=1.5)

	mark_inset(axes[1], axes[2], loc1=3, loc2=4, fc="none", ec='gray', alpha=.5 )

	x=.4; y=.86; dx=.16; dy=.08; ddx=.01; ddy=.02
	axes[2].plot([x, x+dx], [y, y], '-', color='#868585', lw=3, transform=axes[2].transAxes)
	axes[2].plot([x+dx+ddx, x+2*dx+ddx], [y-dy, y-dy], '-', color='#669ac5', lw=3, transform=axes[2].transAxes)
	axes[2].plot([x, x+2*dx+ddx], [y-ddy, y-dy+ddy], '-', color=cs1, lw=2, transform=axes[2].transAxes)
	axes[2].text(x+dx, y+2*ddy, "Post-traitement", transform=axes[2].transAxes, ha='center', va='baseline')

	fig.savefig('../figs/explications_extension.png')
	return fig, axes


def hydro():
	ECQ = np.array( analysis.EC_scen_Q['Beauharnois'] ) + np.array( analysis.EC_scen_Q['lesCedres'] )

	fig, ax = full_flow(None, EC=ECQ)
	fig.savefig("../deliverables/Hydroelectricite/Scenarios_LacOntario.png")
	plt.close()


def lsp():
	fig, ax = full_flow('lsp', analysis.EC_scen_Q['Trois-Rivieres'])
	fig.savefig("../deliverables/Ecosystemes/Scenarios_LSP.png")
	plt.close()


def sorel():
	fig, ax = full_flow('srl', analysis.EC_scen_Q['Sorel'])
	fig.savefig("../figs/Scenarios_Sorel.png")
	fig.savefig("../deliverables/Foncier/Scenarios_Sorel.png")
	plt.close()

def full_flow(site, EC):
	"""Montrer les scenarios de debits, avec la s/rie complete en haut et les scenarios 1 et 2 en bas."""
	from mpl_toolkits.axes_grid1.inset_locator import mark_inset

	def oq(y,m):
		return GLSLutils.qom2date([(y,m)])[0].toordinal()

	oqs = GLSLutils.ordinal_qom


	qR, q1, q2  = analysis.scenarios_Q(site, EC)

	fig = plt.figure(figsize=(8,6))
	fig.subplots_adjust(wspace=.15)
	ax = plt.subplot2grid((2,2), (0,0), colspan=2)
	ax.plot_date(oqs(qR), qR.values, '-', color=cbc, label="Scénario de référence")
	ax.plot_date(oqs(q1), q1.values, '-', color=cs1, label="Scénario 1")
	ax.plot_date(oqs(q2), q2.values, '-', color=cs2, label="Scénario 2")
	ax.set_ylabel("Débit [m³/s]")
	ax.legend(loc='lower center', frameon=False, fontsize='small', bbox_to_anchor=(.5, .98), ncol=3)

	ax1 = plt.subplot2grid((2,2), (1,0), sharey=ax)
	ax1.plot_date(oqs(q1), q1.values, '-', color=cs1)
	ax1.set_ylabel("Débit [m³/s]")
	ax1.set_xticks([oq(y, 1) for y in range(2010, 2071,10)])
	ax1.text(.02, .97, "#1 Chaud et sec\n 2015-2065", size="large", weight='bold', va='top', transform=ax1.transAxes)


	ax2 = plt.subplot2grid((2,2), (1,1), sharey=ax, sharex=ax1)
	ax2.plot_date(oqs(q2), q2.values, '-', color=cs2)
	ax2.text(.02, .97, "#2 Cycle saisonnier amplifié\n 2015-2065", size="large", weight='bold', va='top', transform=ax2.transAxes)

	ax1.set_xlim(oq(2014,1), oq(2066,12))
	mark_inset(ax, ax1, loc1=2, loc2=3, fc="gray", ec='none', alpha=.1 )
	mark_inset(ax, ax2, loc1=3, loc2=4, fc="gray", ec='none', alpha=.1 )

	ax2.yaxis.tick_right()
	ax.tick_params(labelright=True)
	fig.tight_layout()
	return fig, ax


def transport(i):
	"""Comparer les séries observées à la jetée #1 avec les scénarios."""
	import HYDATio
	zc = 5.56

	# Observations - 0 des cartes
	Hd = HYDATio.get_hydat('02OA046', 'H') - zc # Daily
	H = GLSLutils.group_qom(Hd).mean()

	# Scénario REF
	ECL = analysis.EC_scen_L['mtl']
	tsr = analysis.scenario_HR(EC=ECL) - zc


	if i==1:
		# Plot both time series (QOM)
		a = .6
		fig, ax = plt.subplots(1,1, figsize=(10, 5))
		fig.subplots_adjust(left=.07, right=.99)
		bcind = GLSLutils.ordinal_qom(tsr)
		Lobs = ax.plot_date(GLSLutils.ordinal_qom(H), H.values, '-', color='#205e88', lw=0.8, alpha=a,  label='Observations')[0]
		Lbc = ax.plot_date(bcind, tsr.values, '-', color=cbc, lw=1.1, alpha=0.6,  label='Scénario de référence')[0]

		ax.set_xlim(bcind[0], bcind[-1])
		ax.set_ylim(-.6, 3.3)
		ax.set_xlabel("Année")
		ax.set_ylabel("Niveau à la Jetée #1 [m]")
		ax.text(0.02, 1.01, "Moyennes quart-de-mois par rapport au zéro des cartes", fontsize='small', transform=ax.transAxes)
		ax.legend(loc='upper right', frameon=False, fontsize='small')
		return fig, ax

	# Plot both time series (annual mean)
	if i==2:
		a = .6
		fig, ax = plt.subplots(1,1, figsize=(10, 5))
		fig.subplots_adjust(left=.07, right=.99)
		bcind = GLSLutils.ordinal_qom(tsr)
		Hm = H.groupby(level=0).mean()
		Rm = tsr.groupby(level=0).mean()

		Lobs = ax.plot(Hm.index, Hm.values, '-', color='#205e88', marker='o', mec='#205e88', mfc='w', lw=1.5, mew=1.5, alpha=a,  label='Observations')[0]
		Lbc = ax.plot(Rm.index, Rm.values, '-', color=cbc, marker='o', mec=cbc, mfc='w', lw=1.5, mew=1.5, alpha=0.6,  label='Scénario de référence')[0]

		ax.set_xlim(Rm.index[0], Rm.index[-1])

		ax.set_xlabel("Année")
		ax.set_ylabel("Niveau à la Jetée #1 [m]")
		ax.text(0.02, 1.01, "Moyennes annuelles par rapport au zéro des cartes", fontsize='small', transform=ax.transAxes)
		ax.legend(loc='upper right', frameon=False, fontsize='small')
		return fig, ax


	# Distribution (1980-2010)
	if i==3:
		fig, ax = plt.subplots(1,1)
		ax.hist([H.valid().ix[1980:2010].values, tsr.ix[1980:2010]], 20, color=['#205e88', cbc], alpha=.7, label=['Observations', 'Scénario de référence'], linewidth=0)

		ax.legend(loc='upper right', frameon=False, fontsize='small')
		ax.set_xlabel("Niveau à la Jetée #1 [m]")
		ax.set_ylabel("Nb d'occurrences entre 1980 et 2010")
		ax.text(0.02, 1.01, "Moyennes quart-de-mois par rapport au zéro des cartes", fontsize='small', transform=ax.transAxes)

		return fig, ax


def save_transport():
	fig, ax = transport(1)
	fig.savefig('../deliverables/Transport/Jetee1_quartdemois.png')
	plt.close()

	fig, ax = transport(2)
	fig.savefig('../deliverables/Transport/Jetee1_annuel.png')
	plt.close()

	fig, ax = transport(3)
	fig.savefig('../deliverables/Transport/Jetee1_distribution.png')
	plt.close()


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

def graph_dispersion():
	"""Plot the dispersion graph for the relation between flow and levels."""
	# Observations at Pointe-Claire
	s = '02OA039'

	M = HYDATio.get_station_meta(s)
	lat, lon = M['LATITUDE'], M['LONGITUDE']
	ECL = analysis.interpolate_EC_levels(lon, lat)

	Q = analysis.EC_scen_Q['Sorel']

	fig, ax = plt.subplots(1,1, figsize=(6.5,4))
	fig.subplots_adjust(bottom=.15)
	ax.plot(Q, ECL, '-', marker='o', color='k', mfc='w', lw=2, mew=2)

	Hd = HYDATio.get_hydat(s, 'H')
	H = GLSLutils.group_qom(Hd).mean()
	H.index.names = ('Year', 'QTM')

	tsr = analysis.scenario_QR()

	tsr, H = tsr.align(H, 'inner', axis=0)
	ax.plot(tsr, H, '.', color="orange", alpha=.1, ms=2)

	ax.set_xlabel("Débit reconstitué à Sorel [m³/s]")
	ax.set_ylabel("Niveau à Pointe-Claire [m]")
	return fig, ax
#
def compare_MOWAT():
	from scipy.stats import gaussian_kde as kde
	import itertools

	fig, axes = plt.subplots(1,6,figsize=(6.5, 4), sharey=True, facecolor='w')
	fig.subplots_adjust(top=.85)

	# Scenarios from the MOWAT center
	mowat = analysis.MOWAT()


	# Lake Ontario outflow
	ECQ = np.array( analysis.EC_scen_Q['Beauharnois'] ) + np.array( analysis.EC_scen_Q['lesCedres'] )
	our, ou1, ou2 = analysis.scenarios_Q(EC=ECQ)


	ts = (our.ix[1980:2010], ou1, ou2) + mowat

	vmin = min(map(min, ts))
	vmax = max(map(max, ts))

	q = np.linspace(vmin, vmax, 200)

	titles = itertools.cycle(['Référence\n1980-2010', "Scénario 1\nChaud & Sec", "Scénario 2\nAmplification", "Référence\n1900-1989", "Scénario 1\nCCCma 2030", "Scénario 2\nCCCma 2050"])
	dx=.05
	for ax in axes[3:]:
		bb = ax.get_position()
		ax.set_position([bb.x0+dx, bb.y0, bb.width, bb.height])


	for ax, x in zip(axes, ts):
		K = kde(x.values)

		ax.plot(K(q), q, '-', color='k', lw=1.5)

		ax.spines['bottom'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		ax.set_xticks([])
		ax.yaxis.tick_left()
		ax.set_title(titles.__next__(), size='small')

	axes[0].set_ylabel("Débit à la sortie du lac Ontario [m³/s]")
	axes[2].set_xlabel("Fréquence d'occurrence")


	axes[1].text(.5, 1.14, "Ouranos", transform=axes[1].transAxes, weight='bold', ha='center')
	axes[4].text(.5, 1.14, "MOWAT", transform=axes[4].transAxes, weight='bold', ha='center')

	fig.savefig('../figs/comparaison_mowat_ouranos.png', dpi=300)
	return fig, axes









#
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


	fig, axes = plt.subplots(2, figsize=(8,6), sharex=True)
	fig.subplots_adjust(right=.97, left=.09, bottom=.09, hspace=.1, top=.96)
	x = range(12)

	lake = 'lacgreat lakes'; i=0
	ax = axes[0]
	ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
	for (r,f) in analysis.aliases.items():
		ax.plot(x, nbs[lake][r], marker='o', lw=1, mec='none', label=f)
	ax.plot(x, nbs[lake]['obs'], ms=10, lw=2, color='#272727', label='Obs.'    )
	ax.set_ylabel('NBS (Référence) mm/j')

	ax = axes[1]
	ax.set_color_cycle(plt.cm.jet(np.linspace(0,1,10)))
	for (r,f) in analysis.aliases.items():
		ax.plot(x, np.array(nbs[lake][f]) - np.array(nbs[lake][r]), marker='o', lw=1, mec='none', label='{0}/{1}'.format(r,f))
	ax.axhline(0, color='gray', zorder=-1)
	#ax.text(.02, .7, 'Great Lakes CC', ha='left', va='baseline', size='large', weight='bold', color='#272727', transform=ax.transAxes)
	ax.set_xlabel('Mois')
	ax.set_ylabel('ΔNBS')
	ax.set_xticks(x)
	ax.set_xticklabels([m.capitalize() for m in months])
	ax.set_xlim(-.5, 11.5)
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
	from mpl_toolkits.axes_grid1.inset_locator import mark_inset
	from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, DateFormatter



	bc = analysis.scenario_QR()
	s2 = analysis.scenario_Q2()

	fig = plt.figure(figsize=(10,4.5))
	fig.subplots_adjust(left=.1, right=.9, wspace=.15)

	ax = plt.subplot2grid((1,3), (0,0), colspan=2)

	axt = plt.twiny()


	Lbc = axt.plot_date(oqs(bc), bc.values, '-', color=cbc, lw=.6, alpha=.9,  label='Débits reconstitués à Sorel (1953–2012)')[0]

	# Scenario 2
	Ls2 = ax.plot_date(oqs(s2), s2.values, '-', color=cs2, lw=1., label='Débits scénario #2 (2015–2065)')[0]
	ax.set_xlim(oq(2010,1), oq(2069,48))
	ax.set_ylabel('Débit à Sorel [m³/s]')

	ax.legend((Lbc, Ls2), (Lbc.get_label(), Ls2.get_label()), loc='upper right', frameon=False, fontsize='small')

	#ax.set_xlim(dt.datetime(2009,6,1).toordinal(), dt.datetime(2070,6,1).toordinal())
	#axt.set_xlim(dt.datetime(1952,6,1).toordinal(), dt.datetime(2013,6,1).toordinal())
	plt.setp(ax.get_xticklabels(), color=cs2)
	plt.setp(axt.get_xticklabels(), color=cbc)

	axz = plt.subplot2grid((1,3), (0,2))
	axzt = plt.twiny(axz)

	y = 2007
	axzt.plot_date(oqs(bc.ix[y:y+1]), bc.ix[y:y+1].values,  '-', color=cbc, lw=1.5, alpha=.9,)
	axz.plot_date(oqs(s2.ix[y+57:y+58]), s2.ix[y+57:y+58].values,  '-', color=cs2, lw=1.5, alpha=.9,)
	mark_inset(ax, axz, loc1=1, loc2=4, fc="gray", ec='gray', alpha=.2 )
	axz.xaxis.set_major_locator(YearLocator())
	axz.xaxis.set_minor_locator(MonthLocator(interval=2))

	axzt.xaxis.set_major_locator(YearLocator())
	axzt.xaxis.set_minor_locator(MonthLocator(interval=2))
	axz.yaxis.tick_right()
	#axz.set_ylabel('Débit à Sorel [m³/s]')
	axz.annotate('Amplification\ndes minima', xy=(oq(2064, 38), 6700), xytext=(30, 5),\
	  ha='left',
	 xycoords='data', textcoords='offset points', \
	 arrowprops=dict(arrowstyle="->",  connectionstyle="arc3,rad=-.4", color='gray'))
	fig.tight_layout()


	fig.savefig('../figs/scenario2.png')
	return fig

def explain_scenario_2():
	r, f = analysis.NBS_delta()
	bc, s2 = analysis.scenario_2()

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
def foncier():
	fig, ax = plt.subplots(1,1)
	col = plt.cm.RdBu(np.linspace(0,1, 8))
	for reg in ['lsl',]:# 'mtl_lano', 'lsp']:

		T = analysis.get_tesselation(reg)

		features = []
		for i in range(8):
			d = ECio.EC_depth(reg, i+1)
			C = ax.tricontour(T, d, [0,], colors=[col[i],], label=str(i))

	leg = ax.legend(loc='upper right', frameon=False, title='Scénario EC')

	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	return fig, ax

	#
def compare_bases_cases_for_Laura():
	bc_orig = FFio.level_series_QH("srl")
	bc_up = FFio.FF_level('srl', up=True)
	bc_EC_Q = ECio.Q_Sorel('qtm')
	ECL = [ 3.09323328,  3.49145638,  4.1352682 ,  4.74138233,  5.49859576,
		6.38552684,  7.10856182,  8.12003153]

	bc_EC = pd.Series(analysis.weight_EC_levels(ECL, analysis.get_EC_scenario_index(bc_EC_Q)), bc_EC_Q.index)


	if 0:
		fig, ax = plt.subplots(figsize=(11,6))
		fig.subplots_adjust(left=.1, right=.98)

		Lbc = ax.plot_date(util.ordinal_qom(bc_orig), bc_orig.values, '-', color='green', lw=.6, alpha=.9,  label='Base Case original')
		Lbc = ax.plot_date(util.ordinal_qom(bc_up), bc_up.values, '-', color='orange', lw=.6, alpha=.9,  label='Base Case UGLS')
		Lbc = ax.plot_date(util.ordinal_qom(bc_EC), bc_EC.values, '-', color=cbc, lw=.6, alpha=.9,  label='Niveaux reconstitués à Sorel')

		ax.set_ylabel("Niveau à Sorel [m]")
		ax.set_xlim(xmin=dt.datetime(1953,1,1).toordinal())
		ax.legend(fontsize='small', ncol=3, frameon=False )

	if 1:
		fig, ax = plt.subplots(figsize=(8,6))
		x = np.linspace(0,11, 48)
		x = np.arange(1, 49)

		ax.plot(x, bc_orig.ix[1962:1990].groupby(level=1).mean(), color='green', lw=1.5, label='Base Case original')
		ax.plot(x, bc_up.ix[1962:1990].groupby(level=1).mean(), color='orange', lw=1.5, label='Base Case UGLS')
		ax.plot(x, bc_EC.ix[1962:1990].groupby(level=1).mean(), color=cbc, lw=1.5, label= 'Niveaux reconstitués à Sorel')

		ax.legend(fontsize='small', ncol=1, frameon=False )
		ax.set_ylabel("Niveau à Sorel [m]")
		ax.set_xlabel("Quart de mois")

	return fig, ax



def do_synthesis():
	s = analysis.scenarios_H('srl')
	fig, ax = synthesis_scenarios(*s)
	fig.savefig('../rapport/figs/synthese_scenarios.png', bbox_inches='tight', dpi=300)
	plt.close()
#
def synthesis_scenarios(R, S1, S2):
	var = 'Level m'
	r = R.ix[1980:2010]
	s1 = S1.ix[2040:2065]
	s2 = S2.ix[2040:2065]

	rac = r.groupby(level=1).mean()
	s1ac = s1.groupby(level=1).mean()
	s2ac = s2.groupby(level=1).mean()

	x = np.linspace(0,11, 48)

	fig, ax = plt.subplots(1,1, figsize=(6.75, 4))
	lw = 2
	ax.plot(x, rac, '-', color=cbc, lw=lw, label="Référence (1980-2010)")
	ax.plot(x, s1ac, '-', color=cs1, lw=lw, label="Scénario #1 (2040-2065)")
	ax.plot(x, s2ac, '-', color=cs2, lw=lw, label="Scénario #2 (2040-2065)")

	if 1:
		rq = r.groupby(level=1).quantile([.1, .9])
		s1q = s1.groupby(level=1).quantile([.1, .9])
		s2q = s2.groupby(level=1).quantile([.1, .9])

		ax.fill_between(x, rq.ix[:,.1], rq.ix[:,.9], edgecolor=cbc, lw=.5, facecolor=cbc, alpha=.05, clip_on=True)
		ax.plot(x, s1q.ix[:,.1], color=cs1, lw=.5)
		ax.plot(x, s1q.ix[:,.9], color=cs1, lw=.5)
		ax.plot(x, s2q.ix[:,.1], color=cs2, lw=.5)
		ax.plot(x, s2q.ix[:,.9], color=cs2, lw=.5)


	strip(ax)
	plt.setp(ax, xticks=range(12), xlim=(0,11))
	ax.set_xticklabels([m.capitalize() for m in months])
	ax.set_ylabel('Niveau [m]')
	ax.legend(loc='upper right', frameon=False, fontsize='small')

	return fig, ax



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

	fig.subplots_adjust(right=.97, top=.95, bottom=.2)
	strip(ax)

	#ax2 = plt.twiny(ax)
	#ax2.spines["bottom"].set_position(("axes", -.1))
	#make_patch_spines_invisible(ax2)
	#ax2.spines["bottom"].set_visible(True)

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

def plot_flow_level_rel(data):

	fig, ax = plt.subplots(1,1)

	for scen in data.keys():
		ax.plot(data[scen]['q'], data[scen]['l'], '.', label=scen)

	ax.set_xlabel("Débit [m³/s]")
	ax.set_ylabel("Niveau [m]")


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
	pts = np.hstack(ECio.EC_pts().values())
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

def test():
	M = basemap.Basemap(projection = 'omerc',           \
	                     resolution  = 'l',                   \
	                    llcrnrlon  = -43.7,   \
	                    llcrnrlat   = 14.7,    \
	                    urcrnrlon = -4.0,    \
	                    urcrnrlat  = 41.9,    \
	                    lat_2       = 11.0,    \
	                    lat_1       = 45.5,    \
	                    lon_2      = -27.8,   \
	                    lon_1      = -19.9,
						lon_0 =-43.7,
						lat_0=14.7,
						no_rot=True)
	dl = 200000.
	nx = int((M.xmax - M.xmin) / dl) + 1
	ny = int((M.ymax - M.ymin) / dl) + 1
	lonr, latr,x,y= M.makegrid(nx, ny, returnxy=True)
	M.drawcoastlines()
	M.scatter(x.flatten(), y.flatten(),5,marker='o')
	M.drawparallels(np.arange(10,51,10))
	M.drawmeridians(np.arange(-50,1,10))
	M.plot((-19.9, -27.28), (45.5, 11.0), '-', latlon=True)


def clean_mesh_over_map(scen, var='depth', ax=None):
	#m = basemap.Basemap(projection='omerc', resolution='c', lat_0=45.14, lon_0=-73.9, \
#		lon_2=-72.56, lat_2=45.3, lon_1=-73.67,lat_1=45.55, \
		#width=370000, height=170000, no_rot=False)
	if ax is None:
		fig, ax = plt.subplots(1,1,figsize=(8,4))
		cbax = fig.add_axes([.1,.9,.45,.04])
		fig.set_facecolor('w')
		fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
	else:
		fig = ax.get_figure()

	m = basemap.Basemap(projection='omerc', resolution='f', \
		llcrnrlon=-73.9, llcrnrlat=45.14,\
		urcrnrlon=-72.66, urcrnrlat=46.51,\
		lon_1=-74.44, lat_1=45.3,\
		lon_2=-72.2, lat_2=46.7,\
		lat_0=45.14, lon_0=-73.9,\
		no_rot=True,
		ax=ax
		)
	CL = m.drawcoastlines(color=np.array([33,33,33,160])/256.)
	#ST = m.drawstates(color='gray')
	R = m.drawrivers(linewidth=2, color=np.array([178,208,254,128])/256)
	#plt.setp(CL, alpha=.5)

	crt = np.array([73,97,132])/256



	proj = ECio.MTM8()

	cm = haxby
	cm = plt.cm.gist_ncar_r
	#cm = plt.cm.spectral

	X, Y, Z = np.hstack(ECio.EC_pts().values())
	x, y = m(*proj(X,Y, inverse=True))


	D = np.hstack(ECio.EC_depth(scen=scen, var=var if var!='level' else 'depth').values())

	D = np.ma.masked_less_equal(D, 0)
	ps = 4
	S = m.scatter(x, y, c=D, s=ps, linewidths=0, cmap=cm, vmax=20, vmin=1e-2,)# norm=mpl.colors.LogNorm())

	ax.set_autoscale_on(False)
	ax.set_axis_off()

	cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
	if var == 'depth':
		cb.set_label('Profondeur [m]')
		cb.set_label('Depth [m]')
	if var == 'level':
		cb.set_label('Niveau [m]')
	elif var == 'speed':
		cb.set_label('Vitesse [m/s]')

	ax.text(*m(-73.65, 45.52), s="Montreal", ha='center')
	ax.text(*m(-74.1, 45.52), s="Outaouais", ha='left', color=crt)
	ax.text(*m(-72.7, 46.05), s="St-François", ha='left', color=crt)
	ax.text(*m(-73.13, 45.8), s="Richelieu", ha='left', color=crt)

	return m

def plot_depth_map(scen, pts={}, inset=False, ax=None, var='depth'):
	"""
	var : depth, level, speed
	"""
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

	if ax is None:
		fig, ax = plt.subplots(1,1,figsize=(8,4))
		cbax = fig.add_axes([.1,.9,.45,.04])
		fig.set_facecolor('w')
		fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
		singlefig = True
	else:
		fig = ax.get_figure()
		singlefig=False

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

			ax.annotate(str(key), (x,y), (x,y), transform=AR + ax.transData, ha='left', va='bottom', size='large')
			ax.text(x, y, str(key), transform=AR + ax.transData, fontsize=10)

		#ax.legend(fontsize='small')


	pts = np.hstack(ECio.EC_pts().values())
	D = np.hstack(ECio.EC_depth(scen=scen, var=var if var!='level' else 'depth').values())

	X, Y, Z = pts
	D = np.ma.masked_less_equal(D, 0)
	ps = 4
	if var == 'depth':
		S = ax.scatter(X, Y, c=D, s=ps, linewidths=0, transform=AR + ax.transData, cmap=cm, vmax=45, vmin=1e-2, norm=mpl.colors.LogNorm())
	if var == 'level':
		S = ax.scatter(X, Y, c=D+Z, s=ps, linewidths=0, transform=AR + ax.transData, cmap=cm, vmax=30, vmin=20)#, norm=mpl.colors.LogNorm())
	elif var == 'speed':
		S = ax.scatter(X, Y, c=D, s=ps, linewidths=0, transform=AR + ax.transData, cmap=cm, vmin=0, vmax=2)



		# Convert coordinates into lat, long
		#lon, lat = proj(x, y, inverse=True)

		# Convert back into map projection
		#x, y  = m(lon, lat)

		# Construct Delaunay tesselation
		# I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
		#T = analysis.get_tesselation(reg)

		#plt.tricontourf(T, d, 20, mask=ma, vmin=0, vmax=20, cmap=cm)
		#plt.tripcolor(T, d, vmin=0, vmax=20, cmap=cm)

	if singlefig:
		cb = plt.colorbar(S, cax=cbax, extend='max', orientation='horizontal')
		if var == 'depth':
			cb.set_label('Profondeur [m]')
		if var == 'level':
				cb.set_label('Niveau [m]')
		elif var == 'speed':
			cb.set_label('Vitesse [m/s]')


	if inset:
		axins = zoomed_inset_axes(ax, 3.5, loc=4, borderpad=0.3) # zoom =
		if var == 'depth':
			axins.scatter(X, Y, c=D, s=2, cmap=cm, vmax=45, vmin=1e-2, linewidths=0, transform=AR + axins.transData, norm=mpl.colors.LogNorm())
		elif var == 'speed':
			axins.scatter(X, Y, c=D, s=2, cmap=cm, vmax=2, vmin=0, linewidths=0, transform=AR + axins.transData)

		# sub region of the original image
		x1, y1 = 2475000, 4482000
		axins.set_xlim(x1, x1+2.2E4)
		axins.set_ylim(y1, y1+1.1E4)

		mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.7")
		axins.set_xticks([]); axins.set_yticks([])

	return fig, ax

def show_all_depths(var='depth'):
	fig, axes = plt.subplots(4,2, figsize=(11,14))
	fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1, top=1, bottom=.08)
	cbax = fig.add_axes([.1,.06,.8,.025])
	Q = analysis.EC_scen_Q['Sorel']

	for i in range(8):
		f, ax = plot_depth_map(i+1, inset=False, ax=axes.flat[i], var=var)
		axes.flat[i].text(0.1, .9, "#{0}: {1}m³/s".format(i+1, Q[i]), ha='left', va='top', fontsize=14, fontweight='bold', alpha=.8, transform=axes.flat[i].transAxes)

	cb = plt.colorbar(ax.collections[0], cax=cbax, extend='max', orientation='horizontal')

	if var == 'depth':
		cb.set_label('Profondeur [m]')
	elif var == 'speed':
		cb.set_label('Vitesse [m/s]')


	return fig


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
