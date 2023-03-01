import os
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from pyDOE import lhs # Latin Hypercube Sampling
from tqdm import tqdm

import SG_Seven_Decades.util.DF_utils as dutil
"""
PURPOSE: Monte Carlo simulation to estimate uncertainties in internal deformation velocities
SOURCE: based on the TowardsDataScience article by Shuai Gou: https://towardsdatascience.com/performing-uncertainty-analysis-in-three-steps-a-hands-on-guide-9110b120987e

SECTIONS
PHYSICS METHODS- methods defining physical relationships
GEOMETRIC METHODS - methods defining geometric relationships & parameters
UNCERTAINTY PROPAGATION - methods to use in later applications for quantifying parameter estimates & uncertainties
VISCOSITY INVERSION - Invert data from the borehole strain-rate data in Meier (1957) to estimate system effective viscosity
SHAPE FACTOR ESTIMATION - Use Weighted Least Squares (WLS) to estimate valley-fill geometries from geophysical & geodetic data
MC Vint ESTIMATION - Use parameter estimates & uncertainties to estimate glacier internal deformation rates through time.


"""


#### UNCERTAINTY PROPAGATION ROUTINES ####

def norm_lhs(mean_vector,cov_matrix, n_samps = 1000, criterion='maximin'):
	"""
	Compilation of steps from Gou's article* on MC simulations using LHS
	to provide perturbed parameter estimates given a vector of expected parameter values
	and their corresponding covariance matrix

	*TowardsDataScience article by Shuai Gou: 
	https://towardsdatascience.com/performing-uncertainty-analysis-in-three-steps-a-hands-on-guide-9110b120987e
	:: INPUTS ::
	:type mean_vector: numpy.ndarray - n, or n,1 vector
	:param mean_vector: ordered vector of mean parameter estiamtes
	:type cov_matrix: numpy.ndarray - n x n array
	:param cov_matrix: ordered covariance matrix corresponding to entries in mean_vector
	:type n_samps: int
	:para n_samps: Number of target_samples to generate
	:type criterion: str
	:param criterion: See pyDOE.lhs
	"""
	# 0) Detect the number of parameters & do sanity checks
	nfrommv = len(mean_vector)
	nfromcm,mfromcm = cov_matrix.shape
	if nfrommv == nfromcm == mfromcm:
		n_params = nfrommv
	elif nfrommv > nfromcm == nfromcm:
		print('In development, assume dropped COV matrix entries mean 0-(co)variance')
		n_params = nfrommv
	else:
		print('Poorly scaled input distributions, reassess')
		pass
	# 1) Conduct LHS to produce Uniform PDF samples
	uni_samples = lhs(n=n_params,samples=n_samps,criterion=criterion)
	# 2) Conduct inverse transformation sampling to create standard normal distributions
	std_norm_samples = np.zeros_like(uni_samples)
	for i_ in range(n_params):
		std_norm_samples[:,i_] = norm(loc=0,scale=1).ppf(uni_samples[:,i_])
	# 3) Convert standard normals into bivariate normals
	L = np.linalg.cholesky(cov_matrix)
	target_samples = mean_vector + std_norm_samples.dot(L.T)
	return target_samples

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

def estimate_arclength(poly,dx,domain):
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
	Hroot = np.abs(Hroot[(Hroots > min(domain)) & (Hroots < max(domain))])[0]
	HH = np.polynomial.polynomial.polyval(Hroot,polyD.coef)
	return HH

def calc_ShapeFactor(Hi,A,P):
	"""
	Calculate the shape-factor from Nye (1952c) specified as:

	Sf = A / (Hi * P)

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

def model_Xsection(df_bed,df_surf,bed_po,surf_po,dx=10,pfkwargs={'cov':'unscaled','w_pow':-2}):
	"""
	Estimate polynomial fits to bed elevation and surface elevation estimates
	and uncertainties using numpy.polyfit with specified polynomial orders
	for each surface

	Modeling surfaces as:
		 mod, cov = np.polyfit(df_?.index,df_?['uz_m'].values,?_po,w=df_?['sz_m'].values**-2,cov='unscaled')
	Create discretized
	
	:: INPUTS ::
	:type df_bed: pandas.DataFrame
	:param df_bed: Bed elevation data 
					index = lateral positions in meters
					'uz_m' = elevation sample in meters
					'sz_m' = elevation sample uncertainty in meters
	:type df_surf: pandas.DataFrame
	:param df_surf: Ice-surface elevation data. Formatted as df_bed
	:type bed_po: int
	:param bed_po: bed-elevation polynomial fitting order
	:type surf_po: int
	:param surf_po: surface-elevation polynomial fitting order
	:type dx: float
	:param dx: sampling interval in 

	:: OUTPUTS ::
	:rtype OUT: dictionary
	:return OUT: contains model parameter estimates and covariance matrices
	
	"""
	# Conduct polynomial modeling of bed
	bmod,bcov = polyfit_transect(df_bed,bed_po,**pfkwargs)
	# Conduct polynomial modeling of bed
	smod,scov = polyfit_transect(df_surf,surf_po,**pfkwargs)
	OUT = {'bm':bmod,'bCm':bcov,'sm':smod,'sCm':scov}
	return OUT

def est_Sf_lhs(bm,bCm,sm,sCm,n_samps=1000,criterion='maximin'):
	"""
	Conduct estimation of cross-section geometric parameters, considering model-fit
	covariance matrices. Use a Latin Hypercube Sampler (lhs) with niter realizations
	and assume input models have gaussian error.

	:: INPUTS ::
	:type bm: numpy.ndarray
	:param bm: decending power-order polynomial coefficients fit to bed-elevation data
	:type bCm: numpy.ndarray
	:param bCm: model covariance matrix from the polyfit to bed-elevation data
	:type sm: numpy.ndarray
	:param sm: descending power-order polynomial coefficients fit to surface-elevation data
	:type sCm: numpy.ndarray
	:param sCm: model covariance matrix for the polyfit to surface-elevation data 
	"""
	# Prepare Latin Hyper-Cube Samples
	B_samps = norm_lhs(bm,bCm,n_samps=n_samps,criterion=criterion)
	S_samps = norm_lhs(sm,sCm,n_samps=n_samps,criterion=criterion)
	print('Latin hypercube sampled models produced...beginning iterations')
	ind = []; Sfv = []; Av = []; Pv = []; Hv = []
	for L_ in tqdm(range(n_samps)):
		Lbmod = B_samps[L_,:]
		Lsmod = S_samps[L_,:]
		# Conduct polynomial differencing & root analysis
		polyD,roots = get_diffd_polyfit_roots(smod,bmod)
		R0 = roots[0]
		R1 = roots[1]
		# Get analytic cross-sectional area
		Av.append(calc_A(polyD,roots,R0))
		# Get estimated ice-bed interface arclength
		Pv.append(estimate_arclength(polyB,R0,[R0,R1]))
		# Get maximum ice-thickness estimate
		Hv.append(estimate_Ymax(polyD,[R0,R1]))
		# Calculate Shape-Factor
		Sfv.append(calc_ShapeFactor(HH,AA,PP))
		ind.append(L_)
	df_OUT = pd.DataFrame({'Sf':Sfv,'A':Av,'P':Pv,'H':Hv},index=ind)
		
	return df_OUT

#### ALONG-FLOW METHODS
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

def est_slopes(df,df_std,alias={'1948':('Elev48','1948')},wl=300,ws=10,minfrac=0.01):
	"""
	Estimate slopes from surface elevation data
	:: INPUTS ::
	:type df: pandas.DataFrame
	:param df: DataFrame containing surface elevation transects
	:type df_std: pandas.DataFrame
	:param df_std: DataFrame containing surface elevation uncertainties
	:type alias: dictionary
	:param alias: 

	"""
	# Write sampling rate as window step
	dx = ws
	# Get min and max indices
	Io = df.index.min()
	If = df.index.max()
	# Estimate the number of windows
	nwinds = (If - Io + (1 - min_frac)*2*wl)/ws
	# Underestimate by 1, if necessary
	ntotal = np.floor(wl/dx)
	# Assign further offsets to a window-centered basis
	cw_ = 0.0
	# Create a reference vector for window central times
	WINDEX = Io + (min_frac - 0.5)*wl + np.arange(nwinds)*ws

	for k_ in list(alias.keys()):
		k_ind = []; k_m = []; k_b = []; k_vmm = []; k_vbb = []; k_vmb = []; k_frac = [];
    
        ## INNER LOOP ##

        for w_val in tqdm(WINDEX):
            w_s = w_val - (0.5 + cw_)*wl # Starting Index
            w_e = w_val + (0.5 + cw_)*wl # Ending Index
            w_ref = w_val + cw_*wl # Reference Index
            # Fetch Sub-DataFrame view
            IDX = (df.index >= w_s) & (df.index < w_e)
            # Fetch & Subsample DataFrame
            idf = df[IDX].copy()
            # Get series
            iS_u = idf[flds[k_]]
            # Document indices & valid fraction for output
            k_ind.append(w_ref)
            k_frac.append(valid_frac)
            # If the minimum data requirements are met
            if valid_frac >= min_frac:
                # Get independent variable values
                if DTIDX:
                    dx = (iS_u.index[ind] - w_ref).total_seconds()
                elif not DTIDX:
                    dx = (iS_u.index[ind] - w_ref)
                # Get dependent variable values
                dy = iS_u.values[ind]
                # Get dependent variable variances
                # dv = iS_v.values[ind]

                ####### Core Process ##############################
                p0 = 0,np.mean(dy)
                mod,cov = curve_fit(lin_fun,dx,dy,p0)#,sigma=dv**0.5)
                ###################################################

                # mod,cov = np.polyfit(dx,dy,1,w=dv**(-1),cov='unscaled')
                # Save Results
                k_m.append(mod[0]); k_b.append(mod[1]); k_vmm.append(cov[0][0]); k_vbb.append(cov[1][1]); k_vmb.append(cov[0][1])
            # Write nan entries if insufficient data are present
            else:
                k_m.append(np.nan); k_b.append(np.nan); k_vmm.append(np.nan); k_vbb.append(np.nan); k_vmb.append(np.nan)
        df_k = pd.DataFrame({k_+'_m':k_m,k_+'_b':k_b,k_+'_frac':k_frac,\
                             k_+'_vmm':k_vmm,k_+'_vbb':k_vbb,k_+'_vmb':k_vmb},\
                             index=k_ind)
        # Save result
        save_arg = '%s_%s.csv'%(tmp_save_pre,k_)
        print("Processing of %s complete. Saving as: %s"%(k_,save_arg))
        df_k.to_csv(save_arg, header=True, index=True)
        # Concatenate multiple field pairs into a final output
        print('Concatenating results to main output')
        df_OUT = pd.concat([df_OUT,df_k],ignore_index=False,axis=1)

    return df_OUT

#### VISCOSITY INVERSION ####

def glen_flowlaw_zeros(X,B):
	"""
	Formulation of Glen (1955) rheology for ice where n = 3:
	0 = strain_rate - (driving_stress/B)**3
	This allows use of uncertainties on the strain_rate and driving_stress when
	inverting for B using scipy.optimize.curve_fit()
	"""
	eps = X[0]
	tau = X[1]
	return eps - (tau/B)**3

#### INTERNAL DEFORMATION ####

def calc_SIA_Vint(Sf,Hi,alpha,B,nn=3.,rho=916.7):
	"""
	Calculate the internal deformation velocity (Vint) of a glacier under the
	assumptions of the Shallow Ice Approximation (SIA) (Nye, 1952; Hooke, 2005)
	Namely:
	Nonlinear viscous rheology
	All deformation is due to simple shear

	:: INPUTS ::
	:type Sf: float
	:param Sf: [dimless] valley shape-factor
	:type Hi: float 
	:param Hi: [m] maximum ice-thickness (near centerline)
	:type alpha: float
	:param alpha: [rad] local ice-surface slope
	:type B: float
	:param B: [Pa/yr**(1/nn)] Effective viscosity
	:type nn: float
	:param nn: [dimless] Flow-law exponent, generally 3 (Glen, 1955)
	:type rho: float
	:param rho: [kg/m**3] viscous material density, generally 916.7 for glacier ice

	:: OUTPUT ::
	:rtype Vint: float
	:return Vint: [m/yr] Internal Deformation Velocity
	"""
	Vint = (2/(1+nn))*((Sf*rho*9.81*np.sin(alpha))/B)**nn * Hi**(1+nn)
	return Vint


#################################################################################
######################       MCMC PROCESSING SECTION       ######################
#################################################################################

ROOT = os.path.join('..','..','..','..')
TROOT = os.path.join(ROOT,'processed_data','transects')
BROOT = os.path.join(ROOT,'data','MINDSatUW','BOREHOLE')
AApDATA = os.path.join(TROOT,'Transect_AAp_Compiled.csv')
BBpDATA = os.path.join(TROOT,'Transect_BBp_Compiled.csv')
CCpDATA = os.path.join(TROOT,'Transect_CCp_Compiled.csv')
ELE_ERR = os.path.join(TROOT,'Tennant_Menounos_Elev_Err.csv')
BHD_ROOT = os.path.join(ROOT,'data','MINDSatUW','BOREHOLE')

d_alias = {'1948:('}