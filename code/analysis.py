
import numpy as np
import scipy.stats
import pandas as pd
import GLSLio, FFio, ECio, HYDATio, GLSLutils
from imp import reload
from matplotlib import pyplot as plt
reload(GLSLio)

gev = scipy.stats.genextreme

"""
Fan & Fay: Common 29 years time period: 1962-1990
Base case (bc) is meteorological data driven through the GL model, including regulation.
WW and WD cases are the base case to which a delta is applied.
"""

"""Variables tirées du rapport de Bouchard et Morin et Miron et al (2003) pour le Lac St-Louis."""
# Débits des scénarios à Sorel
EC_scen_Q = {'Sorel':[5000, 6500, 8000, 9500, 12000, 14500, 17500, 20500],
		 'LaSalle':[4572, 5740, 6997, 8304, 10102, 11396, 13174, 14531],
		 'MIP':[398, 728, 960, 1142, 1750, 2772, 3824, 5374],
		 'Assomption':[30,32,43,54,148, 332, 502, 550],
		 'Richelieu':[137, 148, 240, 326, 615, 898, 1044, 1100],
		 'Yamaska': [28, 29, 38, 52, 126, 220, 345, 410],
		 'St-Francois':[120, 128, 139, 155, 330, 572, 850, 980],
		 'Trois-Rivieres':[5319, 6843, 8469, 10093, 13227, 16517, 20188, 23554],
		 'Nicolet':[17,19,24,30,76,130,233,380],
		 'Maskinongé':[7,8,14,16,43,105,119,122],
		 'duLoup':[10,11,14,14,37,92,97,107],
		 'Beauharnois': [4345, 5102,5699,6589,7097,7262,7444,7660],
		 'lesCedres': [190,336,383,427,1464,1751,1815,2401],
		 'Ch. Ste-Anne-Vaudreuil': [29, 381, 872, 1237, 1488, 2316, 3751, 4270],
		}

# Niveaux printemps IGLD85
EC_scen_L = {'mtl': [4.29, 4.95, 5.61, 6.30, 7.19, 7.99, 8.8, 9.82],
			 'var':[3.48, 4.2, 4.95, 5.57, 6.37, 7.2, 7.98, 9.06],
			 'srl': [2.96, 3.56, 4.17, 4.74, 5.42, 6.22, 6.92, 8.01],
			 'trois': [2.52, 2.95, 3.55, 4.06, 4.69, 5.53, 6.16, 7.24],}
#             'lsp': [ 2.765531,  3.076017,  3.779904,  4.375879,  5.096382,  6.007973,
#        6.734745,  7.827078]} # LSP calculé à partir du modèle 2D, et non du tableau.
# srl levels don't match with levels interpolated from the 2D model, although their lat lon are pretty close.

# Temps de retour
EC_scen_T = [10000, 70, 3, None, None, 2, 16, 7000]
""" --- """


# Points de contrôle (F&F) - Positions des stations de niveau d'EC
CP = dict(sites = ('mtl', 'var', 'srl', 'trois'),
		  coords = ((-73.5525, 45.5035), (-73.443667, 45.684333), (-73.115667, 46.047167),  (-72.539167, 46.3405 )),
		  names = ('Jetée #1', 'Varennes', 'Sorel', 'Trois-Rivières'))

# 'Lac St-Pierre', (-72.8955, 46.194833), 'lsp',

gcms = {'CGCM2.3': (('afp',), ('afq',)),
		'CGCM3.1': (('aey', 'aez', 'afa', 'aet', 'aev'),
					('afb', 'afc', 'afd', 'aeu', 'aew')),
		'CNRM-CM3': (('agw',), ('ahb',)),
		'ECHAM5': (('agx', 'ahi', 'ahj'), ('agz', 'ahk', 'ahw'))}

aliases = {'aet': 'aeu', 'aev': 'aew', 'aey': 'afb','aez': 'afc','afa': 'afd',\
		 'afp': 'afq', 'agw': 'ahb','agx': 'agz', 'ahi': 'ahk','ahj': 'ahw'}

def scenario_QR(EC=None):
	"""Return the Reference scenario and translate given the EC table. If EC
	is None, flow is at Sorel."""
	ts = ECio.Q_Sorel('qtm')
	ref = ts.reindex(ts.index.truncate(1953,2012)) #57

	if EC is None:
		return ref
	else:
		return translate(ref, EC)

def scenario_Q1(site, EC=None):
	"""Generate scenario WI1 at site and translate given the EC table."""
	if site in ['srl', 'ont', 'lsp']:
		ns = site
	elif site is None and EC is not None:
		ns = 'srl'
	else:
		raise ValueError

	bc = FFio.total_flow(ns, 'bc')
	wd = FFio.total_flow(ns, 'wd')
	s1 = extend_WI1(bc, wd)

	s1 = s1.reindex(s1.index.truncate(after=2065))

	if site is None:
		return translate(s1, EC)
	else:
		return s1

def scenario_Q2(EC=None):
	"""Return the scenario WI2 and translate given the EC table. If EC
	is None, flow is at Sorel."""

	# NBS delta
	r, f = NBS_delta()
	delta = f - r

	# Sorel base series
	ts = ECio.Q_Sorel('qtm')

	# Compute annual cycle
	an = ts.groupby(level=1).mean()

	# Scale the climate change factor in mm³/s with respect to the amplitude.
	# Translate by two weeks the signal
	x = np.linspace(0,11, 48)
	cc = np.interp(x, range(12), delta/r.ptp()) * an.ptp()
	cc = np.roll(cc, 2)

	bc = ts.reindex(ts.index.truncate(1958,2008)) #57
	#bc = ts.ix[1962:1991]; #78

	s2 = apply_delta(shift_mi(bc, 57), cc)

	if EC is not None:
		return translate(s2, EC)
	else:
		return s2

def scenario_HR(EC=None):
	"""Return the water level according to the reference scenario.
	Values are given for Sorel by default if EC is None."""
	if EC is None:
		EC = EC_scen_L['srl']

	q = scenario_QR()
	return translate(q, EC)

def scenario_H1(site, lon=None, lat=None, EC=None, noFFinterp=False):
	"""Return the water level according to the scenario WI1.
	Values are given for site by default if EC is None."""

	if site in {'mtl', 'var', 'srl', 'pcl', 'lsp', 'trois'}:
		bc = FFio.level_series_QH(site, 'bc')
		wd = FFio.level_series_QH(site, 'wd')

	elif EC is not None:
		bc = interpolate_ff_levels(lon, lat, 'bc', EC, noFFinterp)
		wd = interpolate_ff_levels(lon, lat, 'wd', EC, noFFinterp)

	else:
		raise ValueError


	l = extend_WI1(bc,wd)

	return l

def scenario_H2(EC=None):
	"""Return the water level according to the scenario WI1.
	Values are given at Sorel by default if EC is None."""

	if EC is None:
		EC = EC_scen_L['srl']

	q = scenario_Q2()
	return translate(q, EC)

def scenarios_Q(site=None, EC=None):
	"""Return reference, WI1 and WI2 flow series."""
	return scenario_QR(EC), scenario_Q1(site, EC), scenario_Q2(EC)

def scenarios_H(site=None, lon=None, lat=None, EC=None, noFFinterp=False):
	"""Return reference, WI1 and WI2 level series."""
	return scenario_HR(EC), scenario_H1(site, lon, lat, EC, noFFinterp), scenario_H2(EC)

def scenarios_QH(site=None, lon=None, lat=None, ECQ=None, ECH=None, noFFinterp=False):
	"""Return DataFrames with scenarios REF, WI1 and WI2."""
	if ECH is None:
		ECH = EC_scen_L['srl']
	if ECQ is None:
		ECQ = EC_scen_Q['Sorel']

	qr, q1, q2 = scenarios_Q(site, ECQ)
	hr, h1, h2 = scenarios_H(site, lon, lat, ECH, noFFinterp)

	FR = hr.valid().to_frame('Level m')
	FR['Flow m3s'] = qr

	F1 = h1.valid().to_frame('Level m')
	F1['Flow m3s'] = q1

	F2 = h2.valid().to_frame('Level m')
	F2['Flow m3s'] = q2

	return FR, F1, F2
#
def MOWAT():
	"""
	Return lake Ontario outflows corresponding to the BOC and the two (2030 and 2050)
	MOWAT scenarios.

	Notes
	-----
	For their analysis of energy generation, the MOWAT center uses scenarios
	that correspond to -18% and -25% of the Basis of Comparison (BOC) flows for
	the 2030 and 2050 horizons respectively.

	BOC flows are taken from the International St. Lawrence River Board
	of Control and obtained here from
	http://www.glerl.noaa.gov/data/dashboard/data/hydroIO/flows/

	According to the paper "Economic Impacts of Climate Change on the Canadian Great
	Lakes Hydro–Electric Power Producers: A Supply Analysis", the relative change
	factors are applied to each monthly values.

	"""
	mts, ts = GLSLio.OntarioOutflow()
	mts = mts['1900-01':'1989-12']
	return mts, mts*.82, mts*.75
#
def NBS_delta():
	"""Return the climate change NBS delta on a monthly basis."""

	months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
	ref = {}; fut = {}

	for m in months:
		nbs = GLSLio.NBS(m)
		wa = GLSLio.basins_weighted_NBS_average(nbs)
		ref[m] = wa['afa'] #np.mean([wa[a] for a in aliases.keys()])
		fut[m] = wa['afd'] #np.mean([wa[a] for a in aliases.values()])

	r = np.array([ref[m] for m in months])
	f = np.array([fut[m] for m in months])

	return r, f

def translate(ts, ECL):
	"""Compute weights and apply them to translate a series at Sorel somewhere
	else."""
	wi = get_EC_scenario_index(ts)
	return pd.Series(weight_EC_levels(ECL, wi), ts.index)

def extend_WI1(ref, fut, type='+'):

	Gr = ref.groupby(GLSLutils.qom2season)
	Gf = fut.groupby(GLSLutils.qom2season)

	if type == '+':
		delta =  Gf.mean() - Gr.mean()
	elif type == '*':
		delta =  Gf.mean() / Gr.mean()
	else:
		raise ValueError(type)

	# First segment (2015 - 2024)
	s1 = GLSLutils.select_and_shift(ref, before=1979, after=1988, offset=36)

	# Second segment (2025-2039)
	s2 = GLSLutils.select_and_shift(fut, before=2049, after=2063, offset=-24)

	ext = pd.concat([s1,s2])

	# Application of scaling factor
	sf = [scale_delta(delta, date, y1=1977, y2=2055, ym=2025, type=type) for date, val in ext.items()]
	return pd.concat([ext + sf, fut.reindex(fut.index.truncate(after=2065))])

def scale_delta(delta, date, y1=1977, y2=2055, ym=2025, type='+'):
	"""Return the seasonal delta for (year, qom)."""
	s = GLSLutils.qom2season(date)
	d = delta[s]
	y = date[0]
	dy = y2 - y1
	yr = y1 if y < ym else y2

	if type=='*':
		sf = (1 + (d - 1) * (y - yr)/dy)
		raise NotImplementedError ("There was a bug here. I think this code segment was never used. Check before using.")

	elif type=='+':
		sf = d * (y - yr)/ dy

	return sf

def shift_mi(ts, a):
	out = ts[:]

	l0, l1 = out.index.levels
	l0 += a

	out.index.set_levels([l0, l1], True)
	return out

def apply_delta(ts, delta, y0=1975, y1=2055):
	y = np.array(ts.index.get_level_values(0))
	q = np.array(ts.index.get_level_values(1))

	f = (y - y0)/(y1-y0)
	cc = delta[q-1] * f
	return ts + cc

def synthesis_table(R, S1, S2, ann=True, rel=True, var='H'):
	from docx import Document
	import locale
	#locale.setlocale(locale.LC_ALL, 'fr_CA')

	import calendar


	r = R.ix[1980:2010]
	s1 = S1.ix[2040:2065]
	s2 = S2.ix[2040:2065]

	rac = r.groupby(level=1).mean()
	s1ac = s1.groupby(level=1).mean()
	s2ac = s2.groupby(level=1).mean()

	rmm = rac.reshape(12,4).mean(1)
	s1mm = s1ac.reshape(12,4).mean(1)
	s2mm = s2ac.reshape(12,4).mean(1)

	if rel:
		s1mm -= rmm
		s2mm -= rmm

	fmt = '{' + '0:.{0}f'.format('0' if var=='Q' else 1) +  '}'
	fmtr = '{' + '0:{0}.{1}f'.format('+' if rel else '', '0' if var=='Q' else 1)  +  '}'

	document = Document()
	table = document.add_table(rows=4, cols=14 if ann else 13)
	hdr = table.rows[0].cells
	for i in range(12):
		hdr[i+1].text = calendar.month_name[i+1]
		table.rows[1].cells[i+1].text = fmt.format(rmm[i])
		table.rows[2].cells[i+1].text = fmtr.format(s1mm[i])
		table.rows[3].cells[i+1].text = fmtr.format(s2mm[i])
	if ann:
		i = 12
		hdr[i+1].text = 'Ann'
		table.rows[1].cells[i+1].text = fmt.format(rmm.mean())
		table.rows[2].cells[i+1].text = fmtr.format(s1mm.mean())
		table.rows[3].cells[i+1].text = fmtr.format(s2mm.mean())

	document.save('demo_ann{0}.docx'.format('_rel' if rel else ''))

def QM_NBS(n=10):
	"""Seasonal quantile ratios for NBS, computed by driving GCM.

	Corrections computed for each decile.
	"""
	import collections
	rgcm = {}
	for k, vals in gcms.items():
		for v in vals:
			rgcm[v] = k

	q = {}
	for s in ['winter', 'spring', 'summer', 'fall', 'annual']:
		ref = collections.defaultdict(list)
		fut = collections.defaultdict(list)

		q[s] = {}
		nbs = GLSLio.NBS(s, lambda x: x)
		wa = GLSLio.basins_weighted_NBS_average(nbs)

		for g, (R,F) in gcms.items():
			for r, f in zip(R, F):
				ref[g].extend(wa[r].compressed().tolist())
				fut[g].extend(wa[f].compressed().tolist())

		for gcm in ref.keys():
			q[s][gcm] = quantile_means(ref[gcm], n), quantile_means(fut[gcm], n)

	return q

def print_tables():
	import csv, itertools

	# Print the full table of NBS average for each simulation and each lake.
	nbs = GLSLio.NBS()
	wa = GLSLio.basins_weighted_NBS_average(nbs)
	nbs.pop('lacMHG')
	nbs['lactotal'] = wa

	if 0:
		W = csv.writer(open('../rapport/tableaux/sim_average_per_lake.csv', 'w'))
		header = ['']+[lake[3:].capitalize() for lake in nbs.keys()]
		W.writerow(header)

		table = []
		for g, (R, F) in gcms.items():
			al = list(itertools.chain(*zip(R,F)))
			for a in al:
				row = [a,]
				for lake, val in nbs.items():
					row.append(val[a])
				table.append(row)

		W.writerows(table)
		return table

	if 1:
		W = csv.writer(open('../rapport/tableaux/sim_average_per_season.csv', 'w'))
		q = QM_NBS(1)

		header = ['Winter', 'Spring', 'Summer', 'Fall', 'Annual']
		W.writerow([""]+ header)

		for g in gcms.keys():
			row_r = [g,]
			for s in header:
				row_r.append( q[s.lower()][g][0][0] )
			W.writerow(row_r)

			row_f = ['',]
			for s in header:
				row_f.append( q[s.lower()][g][1][0] )
			W.writerow(row_f)

def model_average(x):
	"""Given a NestedDict, find the level at which the simulations are stored
	by alias then replace the individual simulation values by a mean over
	all simulations belonging to the same pilot and splitting the simulations
	between reference and future periods.
	"""
	pass # incomplete
	import ndict
	assert type(x) == ndict.NestedDict
	out = ndict.NestedDict()

	# Find at which level the alias is stored in the dict
	for i in range(x.depth):
		if 'aet' in x.keylevel(i):
			level = i
			break

	for g, aliases in gcms.items():
		for per, al in zip(('ref', 'fut'), aliases):
			av = []
			for it, val in x.walk({level:al}):
				av.append(val)
			it = list(it)
			it.pop(level)
			it.insert(level, g)
			it.insert(level+1, per)
			out.set(it, np.mean(av))

	return out

def quantile_means(x, n=10):
	"""Split the sorted sample in `n` groups and compute their individual mean."""
	sx = np.sort(x)
	s = len(x)

	bins = np.around(np.linspace(0, s, n+1)).astype(int)

	out = []
	count = []
	for i in range(n):
		b1, b2 = bins[i:i+2]
		a = sx[slice(b1, b2)]
		count.append(len(a))
		out.append(a.mean())

	assert sum(count)==s
	return np.array(out)

def get_tesselation(domain, option='A'):
	"""Return the model tesselation in native coordinates.

	Parameters
	----------
	domain : {'lsl', 'lsp', 'mtl_lano'}
	  Domain name.

	"""
	from scipy.spatial import Delaunay
	import matplotlib.tri as tri

	# Get grid coordinates
	x, y, z = ECio.EC_pts(domain)

	# Convert coordinates into lat, long
	# proj = GLSLio.MTM8()
	# lon, lat = proj(x, y, inverse=True)

	# Construct Delaunay tesselation
	# I use scipy's code since the current Matplotlib release delaunay implementation is fragile.
	#D = Delaunay(np.array([x,y]).T)
	T = tri.Triangulation(x, y)# D.vertices)
	A = tri.TriAnalyzer(T)
	dist = np.max(np.sqrt(np.diff(x[T.triangles], axis=1)**2 + np.diff(y[T.triangles])**2), axis=1)

	if option == 'A': # This is what I initially used
		ma = A.get_flat_tri_mask(.2)
		# Masked values
		#area = _triangle_area(x, y, T.triangles)
		ma = ma | (dist > 200)

	elif option == 'B':
		cr = A.circle_ratios()
		ma = (cr < .01) | (dist > 400)
		for i in range(6):
			T.set_mask(ma)
			ma = A.get_flat_tri_mask(.2)

	T.set_mask(ma)

	return T

def _triangle_area(x,y,i):
	"""Return area of triangles formed by vertices (x[i], y[i]).

	Parameters
	----------
	x,y: ndarray (n,)
	  xy coordinates
	i : ndarray (n,3)
	  Indices of triangle vertices.

	Return
	------
	out : ndarray (n,)
	  Triangle area.
	"""
	n, d = np.asarray(i).shape
	assert d == 3

	a = np.zeros(n)
	for k in range(3):
		a += x[i[:,k]] * (y[i[:,(k+1)%3]] - y[i[:,(k+2)%3]])

	return np.abs(a)

def save_convex_hulls():
	"""For all three grids, save the convex hull in a json file."""
	import json
	from scipy.spatial import ConvexHull
	qhull = {}
	for dom in ['lsl', 'lsp', 'mtl_lano']:
		x, y = GLSLio.get_scen(8, dom, ('X', 'Y'))
		pts = np.array([x,y]).T
		qh = ConvexHull( pts )
		qhull[dom] = pts[qh.vertices].tolist()
		#T = get_tesselation(dom)
		#qhull[dom] = np.array([T.x, T.y]).T[T.convex_hull].reshape(-1,2).tolist()


	with open('../analysis/qhulls.json', 'w') as f:
		json.dump(qhull, f)

def get_domain(lon, lat):
	"""Return the domain holding the coordinates."""
	import json
	import matplotlib as mpl

	# Convert coordinates in native grid coordinates
	P = ECio.MTM8()
	x, y = P(lon, lat)

	# Load convex_hulls
	qhulls = json.load(open('../analysis/qhulls.json'))

	for key, poly in qhulls.items():
		P = mpl.path.Path(poly)
		if P.contains_point((x,y)):
			return key

	raise ValueError("Point not in any domain")

def interpolate_EC_levels(lon, lat, scens=None, allow_interpolation=False):
	"""Return the eight levels interpolated from the grid data.

	Parameters
	----------
	lon, lat: float
	  Geographic coordinates
	scens: list
	  Scenarios (1-8) for which to interpolate the level. If None, return the 8 levels.

	"""
	from matplotlib import tri
	import scipy.interpolate
	# Native coordinates
	P = ECio.MTM8()
	x, y = P(lon, lat)

	# Domain identification
	try:
		dom = get_domain(lon, lat)
	except ValueError:
		dom = None # Use all points from all three domains
		if not allow_interpolation:
			raise ValueError("Point is not in any domain. Allow interpolation to interpolate from nearby points.")


	# Interpolate from grid
	X, Y = ECio.get_scen(8, dom, variables=('X', 'Y'))
	pts = np.array([X,Y]).T

	# Get bottom level
	Z = ECio.get_scen(1, dom, variables=('Z',)).T
	if dom:
		I = scipy.interpolate.NearestNDInterpolator(pts, Z)
	else:
		I = scipy.interpolate.LinearNDInterpolator(pts, Z)
	z = I(x, y)[0]

	# Compute depth for each scenario
	if scens is None:
		scens = range(1,9)

	out = []
	for scen in scens:
		D = ECio.get_scen(scen, dom, variables=('PROFONDEUR',)).T
		I.values = D
		out.append( I(x, y)[0] )

	# Add depth to bottom level
	return np.array(out) + z

def save_control_points_EC_levels():
	levels = []
	for c in CP['coords']:
		try:
			levels.append( interpolate_EC_levels(*c) )
		except:
			levels.append(None)
	return levels

def EC_interpolation(lon, lat, flow_at_sorel):
	"""Return the level interpolated from the EC 2D simulations."""
	L = interpolate_EC_levels(lon, lat)
	wi = get_EC_scenario_index(flow_at_sorel)
	levels = interpolate_EC_levels(lon, lat, [int(np.floor(wi)), int(np.ceil(wi))])
	w = 1.-np.mod(wi,1)
	return levels[0] * w + levels[1]*(1-w)

def interpolate_ff_levels(lon, lat, scen='bc', L=None, noFFinterpolation=False):
	"""For a given point, compute the value interpolated from the Fan &
	Fay levels at the control points (Mtl, Sorel, Varennes and
	Trois-Rivieres) and improve the interpolation using the 2D model
	results.

	Parameters
	----------
	lon, lat : floats
	  Geographic coordinates of point of interest.
	scen : {'bc', 'wd', 'ww'}
	  Climate change scenario: base case, warm & dray, warm & wet.

	Returns
	-------
	out : ndarray
	  Series of levels interpolated from Fan & Fay levels and corrected
	  using the 2D simulation from EC.


	Notes
	-----
	1. Interpolate linearly the Fan & Fay levels
	1.1. Find the upstream and downstream control points
	1.2 Compute the weights to apply to upstream and downstream levels
	1.3 Compute the upstream and downstream levels from the F&F relations
	1.4 Interpolate at the chosen location based on the distance to the control points

	2. Correct for non-linear levels
	2.1 Compute the streamflow at Sorel
	2.2 Identify the scenarios above and below the Sorel streamflow
	2.3 Interpolate the scenario level at the point of interest
	2.4 Get the levels at the upstream and downstream control points for all 8 scenarios and interpolate at the point of interest
	2.5 Compute the difference between the linearly interpolated values between the control points and the interpolated 2D depth.
	2.6 Apply the difference to the Fan & Fay series interpolated at the point of interest

	"""
	# Model domain
	try:
		dom = get_domain(lon, lat)
	except ValueError:
		dom = None


	# 2.1 Compute streamflow at Sorel
	FS = FFio.get_flow_sorel(scen)
	FS = FS.reindex(FS.index.truncate(after=FFio.offset[scen]+29))

	# 2.2 Identify the scenarios above and below the Sorel streamflow
	wi = get_EC_scenario_index(FS)
	wi[FS.isnull().values] = 0

	# 2.3 Interpolate the scenario level at the point of interest
	L = L or interpolate_EC_levels(lon, lat)

	if (dom in ['lsl', None]) or noFFinterpolation: #Because no reference level on the left side of the domain.
		print ('lsl')
		return pd.Series(weight_EC_levels(L, wi), FS.index)


	# 1. Fan & Fay linear interpolation
	# ---------------------------------

	# 1.1 Get the upstream and downstream stations from point
	P = ECio.MTM8()
	x, y = P(lon, lat)

	distj = []
	for c in CP['coords']:
		xc, yc = P(*c)
		distj.append(x-xc + 1j*(y-yc) )

	distj = np.array(distj)
	angles = np.angle(distj, True)
	dist = np.abs(distj)

	# Initial algorithm (buggy)
	si = np.argsort(np.abs(dist))[:2]
	up2 = CP['sites'][si.min()]
	dw2 = CP['sites'][si.max()]

	ui = np.nonzero( (np.abs(angles)<90 ))[0]
	di = np.nonzero( (np.abs(angles)>90 ))[0]

	for i in np.argsort(dist):
		if i in ui:
			upstream = CP['sites'][i]
			break
	for j in np.argsort(dist):
		if j in di:
			downstream = CP['sites'][j]
			break
	if (upstream != up2) or (downstream != dw2):
		print( upstream, downstream, '<-', up2, dw2 )

	# 1.2 Compute the weights to apply to upstream and downstream levels
	Iw = dist[j] / np.sum(dist[[j,i]])

	# 1.3 Compute the upstream and downstream levels from the F&F relations
	upL = FFio.level_series_QH(upstream, scen)
	dnL = FFio.level_series_QH(downstream, scen)

	# 1.4 Interpolate at the chosen location based on the distance to the control points
	FF_L = Iw * upL + (1-Iw) * dnL

	# 2. Correct for non-linear levels
	# --------------------------------
	# 2.4 Get the levels at the upstream and downstream control points for all 8 scenarios and interpolate at the point of interest
	EC_L = Iw * np.array(EC_scen_L[upstream]) + (1-Iw) * np.array(EC_scen_L[downstream])

	# 2.5 Compute the difference between the linearly interpolated values between the control points and the interpolated 2D depth.
	deltas = EC_L - L

	# 2.6 Apply the difference to the Fan & Fay series interpolated at the point of interest
	return FF_L + weight_EC_levels(deltas, wi)

def get_EC_scenario_index(flow):
	"""Return a value indicating where with respect to the 8 scenarios the flow
	value stand. The flow is at Sorel.
	"""
	refQ = EC_scen_Q['Sorel']
	wi = np.interp(flow, refQ, range(1,9))
	return wi

def weight_EC_levels(levels, wi):
	"""Interpolate within the 8 EC levels.

	Parameters
	----------
	site : {'mtl', 'var', 'srl', 'trois'}
	  Control point: Montréal, Varennes, Sorel and Trois-Rivières
	wi : ndarray
	  Weighted index. For example, 3.7 means that EC scenario 3 has weight .3 and
	  scenario 4 has weight .7.

	Returns
	-------
	out : ndarray
	  The levels at sites interpolated from the 8 EC levels according to wi.
	"""
	levels = np.asarray(levels)
	w1 = (np.floor(wi) - 1).astype(int)
	w = 1.-np.mod(wi,1)

	out = levels[w1] * w + levels[w1+1]*(1-w)
	out[w1<0] = np.nan
	return out

def check_usines_traitement_in_domain():
	sites = GLSLio.stations_municipales()

	for key, val in sites.items():
		for i in ['1', '2']:
			lat, lon = val['lat'+i], val['lon'+i]
			if lat is None or lon is None:
				val['dom'+i] = None
				continue

			try:
				val['dom'+i] = get_domain(lon, lat)
			except ValueError:
				val['dom'+i] = None

	return sites

def check_usines_usees_in_domain():
	sites = GLSLio.stations_usees()

	for key, val in sites.items():
			lat, lon = val['lat'], val['lon']
			if lat is None or lon is None:
				val['dom'] = None
				continue

			try:
				val['dom'] = get_domain(lon, lat)
			except ValueError:
				val['dom'] = None

	return sites

def checkPointeClaire():
	def degmin2dec(d,m,c):
		"""Convert degrees, minutes.decimals to plain decimals."""
		return int(d) + float(m + '.' + c)/60.

	lat = 45 +  25.24/60
	lon = -73 -  49.40/60

	ECL = interpolate_EC_levels(lon, lat)
	return ECL
	wi = get_EC_scenario_index([7273,])
	return weight_EC_levels(ECL, wi)

def frequential_analysis(ts):
	"""
	Perform a frequential analysis of the annual minima using a
	Log-Pearson type III distribution.

	The LP3 distribution is fitted by the method of L-Moments, using
	Hosking's code.
	"""

	import bayesidf

	# Type checking and annual minimum finding
	if type(ts.index) == pd.MultiIndex:
		assert ts.index.names[0] == 'Year'
		am = ts.groupby(level=0).min()
	elif type(ts.index) == pd.DateTimeIndex:
		assert ts.index.freq == pd.tseries.offsets.Day
		gr = ts.groupby(lambda x: x.year)
		am = gr.min()
		n = gr.count()
		am[n<300] = np.nan

	# Frequential analysis
	y = np.log(am.dropna())
	lm = bayesidf.samlmu(y)
	p = bayesidf.pelpe3(lm)
	mu, sigma, gamma = p

	return scipy.stats.pearson3(gamma, mu, sigma), am.dropna()
