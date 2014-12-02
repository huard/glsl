import graphs, GLSLio, FFio, analysis, GLSLutils, HYDATio, ECio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os, pickle, json
from imp import reload

reload(graphs)
reload(analysis)
reload(GLSLio)


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
def plot_stations_municipales():
	sites = GLSLio.stations_potables()
	pts = {}
	for key, val in sites.items():
		try:
			lon, lat = val[1]
			pts[key]= (lon, lat)
		except:
			pass
	pts.update( dict(zip(analysis.CP['sites'], analysis.CP['coords'])) )

	fig, ax = graphs.plot_depth_map(8, pts=pts, inset=False)
	return fig
#
def plot_stations_usees():
	sites = GLSLio.stations_usees_2()
	pts = {}
	for key, val in sites.items():
		lat, lon = val['lat'], val['lon']
		if lat is None or lon is None:
			continue
		pts[key] = (lon, lat)

	fig, ax = graphs.plot_depth_map(8, pts=pts, inset=False)
	return fig
#
## Senario #1 for Pointe-Claire

def AECOM_potable():

	meta = GLSLio.stations_potables()

	# Interpolate the levels
	ECLpath = '../analysis/potableECL_N.pickle'
	if os.path.exists(ECLpath):
		ECL = pickle.load(open(ECLpath, 'br'))
	else:
		compute_ECL_potable()
		add_ECL_carrieres()
		ECL = pickle.load(open(ECLpath, 'br'))

	out = {'REF':{} ,'WI1':{}, 'WI2':{}}

	ECL = {17: ECL[17]}

	# Scenario #1
	for (key, Ls) in ECL.items():
		for i in [1,2,3]:
			L = Ls.get(i)
			if L is not None:
				lon, lat = meta[key][i]
				out['WI1']['{0}_NO{1}'.format(key, i)] = analysis.interpolate_ff_levels(lon, lat, 'wd', L, noFFinterpolation=key==17)

	# Scenario #2
	for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):
		for (key, Ls) in ECL.items():
			for i in [1,2,3]:
				L = Ls.get(i)
				if L is not None:
					# Scenario weights
					wi = analysis.get_EC_scenario_index(ts)

					# Compute levels
					out[scen]['{0}_NO{1}'.format(key, i)] = pd.Series(analysis.weight_EC_levels(L, wi), ts.index)

	for key in out['REF'].keys():
		fn = '../deliverables/AECOM/Scenarios_niveaux_eau_potable_{0}_IGLD85.xlsx'.format(key)
		EW = pd.ExcelWriter(fn)
		for i, scen in enumerate(['REF', 'WI1', 'WI2']):
			out[scen][key].to_frame(scen + ' m').to_excel(EW, scen)
		EW.close()

	return out

def AECOM_usee():
	meta = GLSLio.stations_usees_2()

	contributions = {
	(8,12,7,26): ["LaSalle",],
	(18,27,22,9,6,5): ["LaSalle", "MIP", "Assomption"],
	(20,25): ["LaSalle", "MIP", "Assomption", "Richelieu"],
	(1,): ["LaSalle", "MIP", "Assomption", "Richelieu", "Yamaska", "St-Francois", "Maskinongé", "duLoup"],
	(14,17):["LaSalle", "MIP", "Assomption", "Richelieu", "Yamaska", "St-Francois", "Maskinongé", "duLoup", "Nicolet"],
	(11,16): ["Ch. Ste-Anne-Vaudreuil",],
	(13,): ["lesCedres",],
	(2,): ["Beauharnois",],
	(23,) : ["lesCedres", "Beauharnois"]
	}

	ECQ = {}
	for key in meta.keys():
		for k,v in contributions.items():
			if key in k:
				ECQ[key] = sum([np.array(analysis.EC_scen_Q[site]) for site in v])
				break

	# Interpolate the levels
	ECLpath = '../analysis/useesECL_N.pickle'
	if os.path.exists(ECLpath):
		ECL = pickle.load(open(ECLpath, 'br'))
	else:
		ECL = {}
		for k, v in meta.items():
			lat, lon = v['lat'], v['lon']
			try:
				ECL[k] = analysis.interpolate_EC_levels(lon, lat).tolist()
			except ValueError:
				ECL[k] = None
		pickle.dump(ECL, open(ECLpath, 'bw'))

	outL = {'REF':{} ,'WI1':{}, 'WI2':{}}
	outQ = {'REF':{} ,'WI1':{}, 'WI2':{}}

	# Scenario #1
	bc, wd = analysis.scenario_1()
	wi = analysis.get_EC_scenario_index(wd)

	for (key, val) in meta.items():
		lon, lat = val['lon'], val['lat']
		L = ECL.get(key)
		if L is not None: outL['WI1'][key] = analysis.interpolate_ff_levels(lon, lat, 'wd', ECL[key])

		Q = ECQ.get(key)
		if Q is not None: outQ['WI1'][key] = pd.Series(analysis.weight_EC_levels(ECQ[key], wi), wd.index)

	# Scenario #2
	for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):
		# Scenario weights
		wi = analysis.get_EC_scenario_index(ts)

		for key in meta.keys():
			# Compute levels
			L = ECL.get(key)
			if L is not None: outL[scen][key] = pd.Series(analysis.weight_EC_levels(L, wi), ts.index)

			# Compute flows
			Q = ECQ.get(key)
			if Q is not None: outQ[scen][key] = pd.Series(analysis.weight_EC_levels(ECQ[key], wi), ts.index)

	for key in outL['REF'].keys():
		fn = '../deliverables/AECOM/Scenarios_niveaux_eau_usees_{0}_IGLD85.xlsx'.format(key)
		EW = pd.ExcelWriter(fn)
		for i, scen in enumerate(['REF', 'WI1', 'WI2']):
			outL[scen][key].to_frame(scen + ' m').to_excel(EW, scen)
		EW.close()

	for key in outQ['REF'].keys():
		fn = '../deliverables/AECOM/Scenarios_debits_eau_usees_{0}.xlsx'.format(key)
		EW = pd.ExcelWriter(fn)
		for i, scen in enumerate(['REF', 'WI1', 'WI2']):
			outQ[scen][key].to_frame(scen + ' m3/s').to_excel(EW, scen)
		EW.close()

	return outL, outQ



def compute_ECL_potable():
	ECL = {}
	meta = GLSLio.stations_potables()
	for k, v in meta.items():
		ECL[k] = {}
		for i in [1,2,3]:
			if meta[k].get(i) is None:
				continue

			lon, lat = meta[k][i]
			try:
				ECL[k][i] = analysis.interpolate_EC_levels(lon, lat).tolist()
			except ValueError:
				pass

	pickle.dump(ECL, open('../analysis/potableECL_N.pickle', 'bw'))

def write_domain_in_usees():
	from openpyxl import Workbook, load_workbook
	filename='../data/AECOM/Calculs wwtp database check.xlsx'
	wb = load_workbook(filename=filename)
	ws = wb['wwtp']
	sites = analysis.check_usines_usees_in_domain()

	ws['Q1'] = 'Domain'
	for key, val in sites.items():
		ws.cell(row=key+1, column=17).value = val['dom']

	wb.save('../data/AECOM/Calculs wwtp database domain check.xlsx')

def write_domain_in_potables():
	from openpyxl import Workbook, load_workbook

	sites = GLSLio.stations_potables()
	L = pickle.load(open('../analysis/potableECL_N.pickle'))

	filename='../data/AECOM/Calculs UTEP.xlsx'
	wb = load_workbook(filename=filename)
	ws = wb['UTEP']

	cols = ['', 'AV', 'AW', 'AX']

	for i in range(1,4):
		ws['{0}{1}'.format(cols[i],1)] = "Domain{0}".format(i)
		for key, v in sites.items():
			if L[key].get(i) is not None:
				lon, lat = L[key][i]
				ws['{0}{1}'.format(cols[i], key+1)] = analysis.get_domain(lon, lat)

	wb.save('../data/AECOM/Calculs UTEP domain check.xlsx')


def add_ECL_carrieres():
	ECLpath = '../analysis/potableECL_N.pickle'
	ECL = pickle.load(open(ECLpath, 'br'))

	# 12 : La Prairie
	ECL[12][1] = [8.3972052587, 9.0669104397, 9.6616692046, 10.2029027485, 11.0188768969, 11.7560161993, 12.5646368075, 13.3103094861]

	# 18 : Montréal Atwater
	ECL[18][1] = [17.5880882, 18.0475934, 18.555371, 18.990474, 19.633782, 20.1145572, 20.7504656, 21.2578254]

	# 6: Candiac
	ECL[6][1] = [8.7882906836, 9.4553778364, 10.046259892, 10.5828876396, 11.3902671625, 12.1181908673, 12.915365757, 13.6494202691]

	pickle.dump(ECL, open(ECLpath, 'bw'))


def Laura():
	import ECio
	bc_EC_Q = ECio.Q_Sorel('qtm')
	ECL = [ 3.09323328,  3.49145638,  4.1352682 ,  4.74138233,  5.49859576,
	6.38552684,  7.10856182,  8.12003153]

	bc_EC = pd.Series(analysis.weight_EC_levels(ECL, analysis.get_EC_scenario_index(bc_EC_Q)), bc_EC_Q.index)

	fr = bc_EC.to_frame('Niveau' + ' m')
	fr['Debit m3s'] = bc_EC_Q
	fr.to_excel('../deliverables/Sorel_Laura.xlsx')



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
	# 15975	  Lac Saint-Pierre	 87L9000			6.537		   9.980				3.443

	# Les données du modèles sont en Niveau Moyen des Mers, ce qui je pense,
	# correspond à l'altitude. Si on veut reporter ces valeurs au datum de la
	# station de mesure, on doit donc soustraire 6.537 des scénarios.

	# Level observations at station
	# 02OC016	SAINT-PIERRE (LAC) AU CURB NO. 2
	# Stations H Lac St-Pierre
	out = {}
	out['WI1'] = {}
	out['WI2'] = {}
	out['REF'] = {}
	s = '02OC016'

	L = HYDATio.get_hydat(s, 'H')
	M = HYDATio.get_station_meta(s)
	lat, lon = M['LATITUDE'], M['LONGITUDE']

	LQ = GLSLutils.group_qom(L).mean()
	LQ.index.names = ["Year", "QTM"]
	F = LQ.to_frame('Level m')
	out['obs'] = F
	F = F.swaplevel(0,1)

	fn = {}
	fn['l'] = '../deliverables/ECOSYSTEMES/Scenarios_niveaux_Lac_St-Pierre_02OC016_IGLD85.xlsx'
	fn['q'] = '../deliverables/ECOSYSTEMES/Scenarios_debits_Lac_St-Pierre.xlsx'

	# What-if scenario #1
	l = FFio.level_series_QH('lsp', 'wd')
	q = FFio.total_flow('lsp', 'wd')
	#fr = dump_xls(q.reindex(q.index.truncate(after=FFio.offset[scen]+29)), l.reindex(l.index.truncate(after=FFio.offset[scen]+29)), os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))
	out['WI1']['q'] = q.reindex(q.index.truncate(after=FFio.offset['wd']+29))
	out['WI1']['l'] = l.reindex(l.index.truncate(after=FFio.offset['wd']+29))

	# What-if Scenario #2
	ECL = analysis.interpolate_EC_levels(lon, lat)
	for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):

		# Scenario weights
		wi = analysis.get_EC_scenario_index(ts)

		# Compute flow at Trois-Rivières (Est-ce qu'on enlève le débit de la Nicolet? p 48 Morin, Bouchard)
		tsLSP = analysis.weight_EC_levels(analysis.EC_scen_Q['Trois-Rivieres'], wi)

		# Compute level at Lac St-Pierre
		lLSP = analysis.weight_EC_levels(ECL, wi)

	#   fn = 'Water_Levels_WhatIf2_Lac_St-Pierre_{0}_IGLD85.xlsx'.format(scen)
	#   fr = dump_xls(pd.Series(tsLSP, ts.index), pd.Series(lLSP, ts.index), os.path.join('..', 'deliverables', 'ECOSYSTEMES', fn))
		out[scen]['q'] = pd.Series(tsLSP, ts.index)
		out[scen]['l'] = pd.Series(lLSP, ts.index)

	for var in ['q', 'l']:
		EW = pd.ExcelWriter(fn[var])
		for i, scen in enumerate(['REF', 'WI1', 'WI2']):
			out[scen][var].to_frame(scen + ' m').to_excel(EW, scen)
		EW.close()
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

def TRANSPORT():
	"""Niveaux a la jetee #1."""

	fn = '../deliverables/Scenarios_niveaux_Jetee_1_mtl.xlsx'
	EW = pd.ExcelWriter(fn)
	out = {}

	# What-if scenario #1
	out['WI1'] = FFio.level_series_QH('mtl', 'wd')

	# What-if scenario #2 at Sorel -> to Mtl
	ECL = analysis.EC_scen_L['mtl']
	for scen, ts in zip(("REF", "WI2"), analysis.scenario_2()):

		# Scenario weights
		wi = analysis.get_EC_scenario_index(ts)

		out[scen] = pd.Series(analysis.weight_EC_levels(ECL, wi), ts.index)

	for i, scen in enumerate(['REF', 'WI1', 'WI2']):
		out[scen].to_frame(scen + ' m').to_excel(EW, scen)
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
	out['WI1'] = wd.reindex(wd.index.truncate(after=2069))

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

def FONCIER2():
	from osgeo import ogr
	import fiona
	from matplotlib import pyplot as plt
	import shapely

	for reg in ['lsl', 'mtl_lano', 'lsp']:

		T = analysis.get_tesselation(reg)

		features = []
		for i in range(8):
			d = ECio.EC_depth(reg, i+1)
			C = plt.tricontour(T, d, [0,])
			lc = C.collections[0] # line collection
			seg = lc.get_segments()
			plt.close()

			features.append( {'geometry': {'type': 'MultiLineString',
							 'coordinates': seg},
							'properties': { 'SCEN_ID': str(i+1), \
											'FLOW':analysis.EC_scen_Q['Sorel'][i]}})


		with fiona.open('../deliverables/Foncier/GLSL_contour_{0}.tab'.format(reg), 'w',\
						crs={'no_defs': True, 'init': 'epsg:32188'},\
						driver="MapInfo File",\
						schema={'geometry':'MultiLineString',\
							   'properties':{'SCEN_ID': 'str',\
											 'FLOW': 'float:16'}}) as sink:

			sink.writerecords(features)

def FONCIER3():

	fn = '../deliverables/Foncier/Scenarios_debits_Sorel.xlsx'
	EW = pd.ExcelWriter(fn)

	scens = analysis.scenarios_Sorel()
	k = "Flow m3s"
	for scen, ts in zip(["REF", "WI1", "WI2"], scens):
		ts[k].to_frame(k).to_excel(EW, scen)

#	qbc.to_frame("REF + m3s").to_excel(EW, 'REF')#
	#wd.to_frame("WI1 + m3s").to_excel(EW, 'WI1')
	#q2.to_frame("WI2 + m3s").to_excel(EW, 'WI2')

	EW.close()



def HYDRO():
	"""Scénario de débit pour Beauharnois et Les Cedres."""


	fn = '../deliverables/Scenarios_debit_lac_Ontario.xlsx'
	EW = pd.ExcelWriter(fn)
	tmp = {}

	# 1
	wd = FFio.FF_flow('ont', 'wd', FFio.offset['wd']	) #check annees
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
