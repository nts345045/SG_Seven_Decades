import numpy as np
import pandas as pd

#### GEOMETRIC MODEL METHODS ####

def polyfit_transect(df_,po,cov='unscaled',w_pow=-2):
	"""
	Wrapper for numpy.polyfit to use on vertical elevation data and uncertainties
	:: INPUTS :: 
	:type df_: pandas.DataFrame
	:param df_: DataFrame with fields: 	uz_m - mean surface elevations in meters
										sz_m - stdev surface elevations in meters
	:type po: int
	:param po: Polynomial order to use for fitting
	:type cov: str
	:param cov: input kwarg for 'cov' in numpy.polyfit
	:type w_pow: int
	:param w_pow: weight power too use for data in 'sz_m' when passed to kwarg 'w' in numpy.polyfit

	:: OUTPUTS ::
	:rtype imod: numpy.ndarray
	:return imod: descending power ordered polynomial coefficients
	:rtype icov: numpy.ndarray
	:return icov: model covariance matrix output from numpy.polyfit
	"""
	IDX = (df_['uz_m'].notna()) & (df_['sz_m'].notna)
	Xi = df_[IDX].values
	Zu = df_[IDX]['uz_m'].values
	Zs = df_[IDX]['sz_m'].values
	imod,icov = np.polyfit(Xi,Zu,po,w=Zs**w_pow,cov=cov)
	return imod,icov

def get_diffd_polyfit_roots(poly1,poly2,rules=['real','positive'],abs_val=True):
	"""
	Use Numpy root-finding algorithms on polynomials
	to find the intercepts of two polynomials

	:type poly1: numpy.ndarray
	:param poly1: polynomial coefficients with highest powers first (per np.polyfit output `p`)
					This should be the higher-valued polynomial of interest (surface)
	:type poly2: numpy.ndarray
	:param poly2: polynomial coefficients with highest powers first (per np.polyfit output `p`)
					This should be the lower-valued polynomial of interest (bed)
	:type rules: list of strings (even if only 1 option is given) or False
	:param rules: dictate what type of sub-selection of roots to return
	:type abs_val: Bool
	:param abs_val: return output roots as absolute values?

	:: OUTPUTS ::
	:rtype pD: numpy.polynomial.Polynomial
	:return pD: ascending-power-ordered polynomial coefficients arising from (poly1 - poly2)[::-1]
	:rtype OUT: numpy.ndarray
	:return OUT: roots with 'rules' and 'abs_val' applied
	"""
	# Get ascending-power coefficients formatted in np.polynomial format for root-finding
	p1 = np.polynomial.Polynomial(poly1[::-1])
	p2 = np.polynomial.Polynomial(poly2[::-1])
	# Difference polynomials
	polyD = (p1 - p2)
	# Find roots of differenced polynomia
	roots = polyD.roots()
	if 'real' in rules:
		if 'positive' in rules:
			IND = (np.isreal(roots))&(roots > 0)
		else:
			IND = np.isreal(roots)
		OUT = roots[IND]
	else:
		OUT = roots

	if abs_val:
		return polyD, np.abs(OUT)
	else:
		return polyD, OUT

#### ACROSS-FLOW METHODS ####
def calc_A(polyD,roots,left_root):
	"""
	Calculate the cross-sectional area by integrating the differenced polynomial between roots, 
	starting from QC'd left root
	:: INPUTS ::
	:type polyD: numpy.polynomial.Polynomial
	:param polyD: differenced polynomial coefficients, with ascending power order
	:type roots: numpy.ndarray
	:param roots: array of real-valued roots associated withpolyD
	:type left_root: float
	:param left_root: root value to use as the lower bound of integration

	:: OUTPUT ::
	:rtype AA: float
	:return AA: cross-sectional area with implied units from polyD
	"""
	AA = np.diff(np.polynomial.polynomial.polyval(roots,polyD.integ(lbnd=left_root).coef))[0]
	return AA

def estimate_arclength(poly,dx,xl=0,xr=2000):
	"""
	Do a trapezoidal estimation of arclength for a polynomial
	:: INPUTS ::
	:type poly: numpy.polynomial.Polynomial
	:param poly: polynomial to assess
	:type dx: float
	:param dx: x-domain discretization 
	:type domain: array-like
	:param domain: 2-element array defining the bounds of the x-domain to assess
	
	:: OUTPUT ::
	:rtype PP: float
	:return PP: arclength estimate
	"""
	vx = np.arange(xl,xr,dx)
	vy = poly(vx)
	PP = np.trapz(np.sqrt(1 + np.gradient(vy,vx)**2),vx)
	return PP

def estimate_Ymax(polyD,domain):
	"""
	Get maximum polynomial value between two specified input values
	:: INPUTS ::
	:type polyD: numpy.polynomial.Polynomial
	:param polyD: differenced polynomial coefficients, with ascending power order
	:type domain: array-like
	:param domain: 2-element array defining the bounds of the x-domain to assess
	"""
	Hroots = polyD.deriv().roots()
	Hroot = np.abs(Hroots[(Hroots > min(domain)) & (Hroots < max(domain))])[0]
	HH = np.polynomial.polynomial.polyval(Hroot,polyD.coef)
	return HH

def calc_ShapeFactor(Hi,A,P):
	"""
	Calculate the shape-factor from Nye (1965) in the formulation presented
	in Hooke (2005):

	Sf = A / (Hi * P)

	Which plugs into the driving stress equation of the form:
	\tau_d = Sf rho g H sin(\alpha)

	FOR REFERENCE:
	Nye (1965) form is:
	Sf = A / P
	\tau_d = Sf rho g sin(\alpha)

	Note: inputs must either be equally-scaled vectors or coefficient-like. Mixes permitted

	:: INPUTS ::
	:type Hi: float or array-like
	:param Hi: Maximum / centerline ice-thickness
	:type A: float or array-like
	:param A: ice-filled cross-sectional area
	:type P: float or array-like
	:param P: Ice-rock contact perimeter

	:: ROUTPUT ::
	:rtype Sf: float or array-like
	:return Sf: Shape Factor estimate(s)
	"""
	return A / (Hi * P)

#### ALONG-FLOW METHODS ####
def linfun(x,m,b):
	"""
	Linear function: y = mx + b
	"""
	return x*m + b

def m2theta(m):
	"""
	Slope to surface angle conversion
	"""
	return (180/np.pi)*np.arctan(m)


def est_slopes(S_mean,S_std,wl=300,ws=10,minsamp=3,w_pow=-1,pfkw={'deg':1,'cov':'unscaled'}):
	"""
	Estimate slopes from surface elevation data
	:: INPUTS ::
	:type S_mean: panda.Series
	:param S_mean: Spatially indexed series containing a mean surface elevation transect
	:type df_std: pandas.DataFrame
	:param df_std: DataFrame containing surface elevation uncertainties
	:type alias: dictionary
	:param alias: 

	"""
	# Get min and max indices
	Io = S_mean.index.min()
	If = S_mean.index.max()
	# Estimate the number of windows
	nwinds = (If - Io)/ws
	# Underestimate by 1, if necessary
	ntotal = np.floor(wl/ws)
	# Assign further offsets to a window-centered basis
	cw_ = 0.0
	# Create a reference vector for window central times
	WINDEX = Io + np.arange(nwinds)*ws

	k_ind = []; k_m = []; k_b = []; k_vmm = []; k_vbb = []; k_vmb = []; k_count = []; k_alpha = []; k_vaa = []; k_vab= []

	## INNER LOOP ##

	for w_val in WINDEX:
		w_s = w_val - 0.5*wl # Starting Index
		w_e = w_val + 0.5*wl # Ending Index
		w_ref = w_val + cw_*wl # Reference Index
		# Fetch Sub-DataFrame view
		IDX = (S_mean.index >= w_s) & (S_mean.index < w_e) & S_mean.notna() & S_std.notna()
		# Fetch & Subsample Series
		dy = S_mean[IDX].values
		ds = S_std[IDX].values
		dx = S_mean[IDX].index - w_ref
		valid_count = len(dy)
		# Document indices & valid fraction for output
		k_ind.append(w_ref)
		k_count.append(valid_count)
		# If the minimum data requirements are met
		if valid_count >= minsamp:

			####### Core Process ##############################
			mod,cov = np.polyfit(dx,dy,w=ds**w_pow,**pfkw)
			###################################################

			# mod,cov = np.polyfit(dx,dy,1,w=dv**(-1),cov='unscaled')
			# Save Results
			k_m.append(mod[0]); k_b.append(mod[1]); k_vmm.append(cov[0][0]); k_vbb.append(cov[1][1]); k_vmb.append(cov[0][1]);
			k_alpha.append(m2theta(mod[0])); k_vaa.append(m2theta(cov[0][0]**0.5)**2); k_vab.append(m2theta(cov[0][1]))
		# Write nan entries if insufficient data are present
		else:
			k_m.append(np.nan); k_b.append(np.nan); k_vmm.append(np.nan); k_vbb.append(np.nan); k_vmb.append(np.nan); 
			k_alpha.append(np.nan); k_vaa.append(np.nan); k_vab.append(np.nan)
	df_OUT = pd.DataFrame({'m':k_m,'b':k_b,'deg':k_alpha,'vdegdeg':k_vaa,'vdegb':k_vab,'n_obs':k_count,'vmm':k_vmm,'vbb':k_vbb,'vmb':k_vmb},\
						   index=k_ind)

	return df_OUT






# def est_slopes_DF(df,df_std,alias={'1948':('Elev48','1948')},wl=300,ws=10,minfrac=0.01):
# 	"""
# 	Estimate slopes from surface elevation data
# 	:: INPUTS ::
# 	:type df: pandas.DataFrame
# 	:param df: DataFrame containing surface elevation transects
# 	:type df_std: pandas.DataFrame
# 	:param df_std: DataFrame containing surface elevation uncertainties
# 	:type alias: dictionary
# 	:param alias: 

# 	"""
# 	# Write sampling rate as window step
# 	dx = ws
# 	# Get min and max indices
# 	Io = df.index.min()
# 	If = df.index.max()
# 	# Estimate the number of windows
# 	nwinds = (If - Io + (1 - min_frac)*2*wl)/ws
# 	# Underestimate by 1, if necessary
# 	ntotal = np.floor(wl/dx)
# 	# Assign further offsets to a window-centered basis
# 	cw_ = 0.0
# 	# Create a reference vector for window central times
# 	WINDEX = Io + (min_frac - 0.5)*wl + np.arange(nwinds)*ws

# 	for k_ in list(alias.keys()):
# 		k_ind = []; k_m = []; k_b = []; k_vmm = []; k_vbb = []; k_vmb = []; k_count = [];
	
# 		## INNER LOOP ##

# 		for w_val in tqdm(WINDEX):
# 			w_s = w_val - (0.5 + cw_)*wl # Starting Index
# 			w_e = w_val + (0.5 + cw_)*wl # Ending Index
# 			w_ref = w_val + cw_*wl # Reference Index
# 			# Fetch Sub-DataFrame view
# 			IDX = (df.index >= w_s) & (df.index < w_e)
# 			# Fetch & Subsample DataFrame
# 			idf = df[IDX].copy()
# 			# Get series
# 			iS_u = idf[flds[k_]]
# 			# Document indices & valid fraction for output
# 			k_ind.append(w_ref)
# 			k_count.append(valid_frac)
# 			# If the minimum data requirements are met
# 			if valid_frac >= min_frac:
# 				# Get independent variable values
# 				if DTIDX:
# 					dx = (iS_u.index[ind] - w_ref).total_seconds()
# 				elif not DTIDX:
# 					dx = (iS_u.index[ind] - w_ref)
# 				# Get dependent variable values
# 				dy = iS_u.values[ind]
# 				# Get dependent variable variances
# 				# dv = iS_v.values[ind]

# 				####### Core Process ##############################
# 				p0 = 0,np.mean(dy)
# 				mod,cov = curve_fit(lin_fun,dx,dy,p0)#,sigma=dv**0.5)
# 				###################################################

# 				# mod,cov = np.polyfit(dx,dy,1,w=dv**(-1),cov='unscaled')
# 				# Save Results
# 				k_m.append(mod[0]); k_b.append(mod[1]); k_vmm.append(cov[0][0]); k_vbb.append(cov[1][1]); k_vmb.append(cov[0][1])
# 			# Write nan entries if insufficient data are present
# 			else:
# 				k_m.append(np.nan); k_b.append(np.nan); k_vmm.append(np.nan); k_vbb.append(np.nan); k_vmb.append(np.nan)
# 		df_k = pd.DataFrame({k_+'_m':k_m,k_+'_b':k_b,k_+'_frac':k_count,\
# 							 k_+'_vmm':k_vmm,k_+'_vbb':k_vbb,k_+'_vmb':k_vmb},\
# 							 index=k_ind)
# 		# Save result
# 		save_arg = '%s_%s.csv'%(tmp_save_pre,k_)
# 		print("Processing of %s complete. Saving as: %s"%(k_,save_arg))
# 		df_k.to_csv(save_arg, header=True, index=True)
# 		# Concatenate multiple field pairs into a final output
# 		print('Concatenating results to main output')
# 		df_OUT = pd.concat([df_OUT,df_k],ignore_index=False,axis=1)

# 	return df_OUT