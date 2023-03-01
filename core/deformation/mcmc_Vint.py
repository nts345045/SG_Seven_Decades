import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy

import SG_Seven_Decades.util.mcmc_util as mcmc
import SG_Seven_Decades.util.DF_utils as dutil
import SG_Seven_Decades.util.transect_util as tutil
"""
To-Do: Get station names into surface transects
	   Directly read-in f0 estimates from HV processing outputs
"""


#### PARAMETER CONTROL SECTION ####

#### PHYSICAL CONSTANTS ####
fWE = 0.91 				# [m m-1] Water Column Height / Ice Column Height (also sets density for ice)
Vs_u = 1769.			# [m s-1] Mean ice shear-wave velocity
Vs_v = 168.**2 			# [m s-1] Standard deviation of ice shear-wave velocity
gg = 9.81 				# [m s-2] Gravitational acceleration near Earth's surface
mpft = 1./3.28    		# [m ft-1] Unit conversion meters per foot
spa = 3600*24*365.24	# [s a-1] Seconds per year (actual orbital period to 2 decimals)
rpd = np.pi/180			# [rad deg-1] Radians per degree



## STEP 0 - PreProcessing ##
M_GAU_coef = 1.96 	# [coefficient] - sigma bound to assign to min/max strain-rate estimates
DG_GAU_coef = 1 	# [coefficient] - sigma bound to consider the min/max data when estimating the StDev for data
TM_GAU_coef = 1 	# [coefficient] - sigma bound to consider the reported uncertainties from Tennant & Menounos (2013) Table 3
BBp_edges = [300,1650] # [m] - Hard set edge filter to prevent inclusion of rock-elevation data in estimating ice-surfaces
CCp_edges = [450,1650] # [m] - Hard set edge filter to prevent inclusion of rock-elevation data in estimating ice-surfaces
BBp_offset = 1060 # [m] Apply a lateral offset to have valley bottom at x~0
CCp_offset = 1080 # [m] Apply a lateral offset to have valley bottom at x~0

## STEP 1 - Transect Surface Modeling ##
wl = 600		# [m] Length of window for surface elevation smoothing & slope estimation
ws = 10 		# [m] Step size for windows ^^
BBpBo = 3 		# [pow] Model order to use for valley geometry "bed" in B-B'
CCpBo = 3 		# [pow] Model order to use for valley geometry "bed" in C-C'
pSo = 1 		# [pow] Model order to use for ice-surface geometries
xvect = np.arange(-500,2010,10)

## STEP 2 - Effective Viscosity Inversion ##
a54 = 4.41*rpd    # [rad] Surface slope measured near 1952-1954 borehole
Sf54 = 1		# [dimless] Shape-factor to use for deformation
			#           We assume 1 here due to the shallow nature of the borehole

## STEP 3 - Geometry Processing & Vint Estimation ##
n_mcmc = 1000	# [dimless]


def quarter_wavelength(f0,vf0,Vs=3451/1.95,vVs=(168/1.95)**2):
	"""
	Quarter wavelength approximation of thickness from HV
	Bard & Bouchon (1980a,b) with uncertainty propagation
	:: INPUTS ::
	:param f0: Fundamental frequency [Hz]
	:param Vs: Shear-wave velocity [length-Hz]
	:param vf0: Fundamental frequency variance [Hz**2]
	:param vVs: Shear-wave velocity variance [length**2-Hz**2]

	:: OUTPUTS ::
	:return Hu: Thickness estimate expectation [length]
	:return Hv: Thickness estimate variance [length**2]
	"""
	Hu = Vs/(4.*f0)
	Hv = (Hu**2)*((vVs/Vs**2) + (vf0/f0**2))
	return Hu,Hv

def flowlaw(tau,B):
	"""
	Glen (1955) type flow-law
	"""
	return (tau/B)**3


def driving_stress(rho,g,z,Sf,alpha):
	"""
	Hooke (2005) formatted driving stress of Nye (1965).
	Note that alpha must be in radians
	"""
	return rho*g*z*Sf*np.sin(alpha)


def Vint(rho,g,H,Sf,alpha,B):
	return 0.5*H*(driving_stress(rho,g,H,Sf,alpha)/B)**3



### MAP DATA ###
# Overall Root Path
ROOT = os.path.join('..','..','..','..')
### DATA INPUTS ###
# Transect Data Root Path
TROOT = os.path.join(ROOT,'data','MINDSatUW','GIS','Transects')
# Borehole Deformation Data Root Path
BROOT = os.path.join(ROOT,'data','MINDSatUW','BOREHOLE')
# Along-Flow Transect Elevation Data File
AApDATA = os.path.join(TROOT,'??*','*AAp*.csv')
BBpDATA = os.path.join(TROOT,'??*','*BBp*.csv')
CCpDATA = os.path.join(TROOT,'??*','*CCp*.csv')
# Interpreted recently Exposed Bedrock Data
BBpBRD = os.path.join(TROOT,'BBp_Exposed_Bedrock_v2.csv')
CCpBRD = os.path.join(TROOT,'CCp_Exposed_Bedrock_v2.csv')
# Elevation Error Reference Table from Tennant & Menounos (2013)
ELE_ERR = os.path.join(TROOT,'TM13','Tennant_Menounos_Table2_3_merge.csv')
# Borehole deformation data from Meier (1957)
BHD_ROOT = os.path.join(ROOT,'data','MINDSatUW','BOREHOLE','*.csv')

### DATA OUTPUTS ###
OTROOT = os.path.join(ROOT,'processed_data','transects')
ODROOT = os.path.join(ROOT,'processed_data','deformation')


#########################################
#### STEP 0: Load & Pre-Process Data ####
#########################################
print('Starting STEP 0 - Data Pre-Processing')
## Step 0a: Load individual extracted profiles 
flist_AAp = glob(AApDATA)
flist_BBp = glob(BBpDATA)
flist_BBp.append(BBpBRD)
flist_CCp = glob(CCpDATA)
flist_CCp.append(CCpBRD)
flists = {'A':flist_AAp,'B':flist_BBp,'C':flist_CCp}
flist_BHD = glob(BHD_ROOT)

# Load Tennant & Menounos 
df_TM_err = pd.read_csv(ELE_ERR,index_col=0,parse_dates=['Date'])

# Iterate Across Transects
IDF_dict = {}
for k_ in flists.keys():
	flist = flists[k_]
	# Create holder DataFrame for transect elements
	IDF = pd.DataFrame()
	for f_ in flist:
		# Get source tag from subdirectory name
		f_dir,f_fil = os.path.split(f_)
		f_src = os.path.split(f_dir)[-1]
		# Read in raw data
		idf = pd.read_csv(f_,index_col=0)

		## Do data-source-specific pre-processing ##
		# Surveys From 2017 / 2019 - do quarter wavelength processing here
		if f_src == 'SGGS':
			if k_ in 'C':
				s_yr = 2019
				i_flds = ['Elev19']#,'Estd19']
				# Set re-labeling dictionary
				i_rname = {'Elev19':'Surf18_mean'}
				# Assign generalized uncertainty from continuous data
				surf_std = pd.Series(np.ones(len(idf),)*0.005,name='Surf18_std',index=idf.index)
			elif k_ == 'B':
				s_yr = 2017
				i_flds = ['Elev17','Estd17']
				i_rname = {'Elev17':'Surf18_mean'}
				surf_std = pd.Series(np.ones(len(idf),)*1.2,name='Surf18_std',index=idf.index)
				surf_std.name = 'Surf18_std'
			elif k_ in 'A':
				s_yr = 2019
				i_flds = ['Elev19']#,'Estd19']
				# Set re-labeling dictionary
				i_rname = {'Elev19':'Surf18_mean'}
				# Assign generalized uncertainty from continuous data
				errs = np.ones(len(idf))
				errs[idf['Type']=='SS19'] = 0.005
				errs[idf['Type']=='SG19'] = 0.005
				errs[idf['Type']=='SS17'] = 1.2
				surf_std = pd.Series(errs,name='Surf18_std',index=idf.index)
			# Conduct ice-bed interface estimation 
			iHu,iHv = quarter_wavelength(idf['f0_mean'],idf['f0_std']**0.5)
			# Create ExTRA dATAfRAME - bed elevation and uncertainty (no propagation of surface uncertainty here, this is done later)
			xdf = pd.DataFrame({'Bed18_mean':(idf[i_flds[0]][idf['Mask']==1] - iHu[idf['Mask']==1]).values,\
								'Bed18_std':iHv[idf['Mask']==1].values**0.5},\
								index=idf[idf['Mask']==1].index)
			xdf = pd.concat([xdf,surf_std],axis=1,ignore_index=False)
		# Data from Tennant & Menounos (2013) - merge in 
		elif f_src == 'TM13':
			s_yr = f_fil.split('_')[1]
			i_flds = idf.filter(like='Elev').columns
			# Get uncertainties from look-up table
			err = np.ones(len(idf))*df_TM_err[df_TM_err['Year']==int(s_yr)]['m'].values/fWE
			i_rname = {i_flds[0]:'Surf%2s_mean'%(s_yr[-2:])}
			# Create ExTRA dATAfRAME - elevation uncertainties from Tennant & Menounos (2013) Table 2
			xdf = pd.DataFrame({'Surf%2s_std'%(s_yr[-2:]):err},index=idf.index)
		# Data from Danielson & Gesch (2011)
		elif f_src == 'DG11':
			s_yr = 2010
			i_flds = ['Elev2010_mean']
			# Set re-labeling dictionary
			i_rname = {i_flds[0]:'Surf10_mean'}
			# Get gaussian estimate of 1-sigma based on assumed min-max gaussian representation
			sig_est = (idf['Elev2010_max'] - idf['Elev2010_min'])/(DG_GAU_coef*2)
			# Create std DataFrame as Extra
			xdf = pd.DataFrame({'Surf10_std':sig_est.values},index=idf.index)
		# Bedrock Elevation Profiles
		elif f_src == 'Transects':
			# Get exposed bedrock elevation data sample standard deviations for sites flagged as "recently exposed"
			BR_std = idf.copy()[idf['Mask']==1][idf.filter(like='Elev').columns].std(axis=1)
			# Get mean elevations for these points
			BR_mea = idf.copy()[idf['Mask']==1][idf.filter(like='Elev').columns].mean(axis=1)
			# Filter out 2017/2019 ice-surface elevation data in the mean
			i_flds = ['ExposedBed_std']
			xdf = pd.DataFrame({'ExposedBed_mean':BR_mea[BR_std.notna()].values},index=BR_std[BR_std.notna()].index)
			idf = pd.DataFrame({'Mask':idf['Mask'].values,'ExposedBed_std':BR_std},index=idf.index)
			i_rname = {}
		### Make filters
		# Filter for Along-Flow Transect Surface Elevation Data
		if f_src == 'Transects' or k_ == 'A':
			filt = idf['Mask']==1
		# Filter for Upper Sector 
		elif k_ == 'B' and f_src != 'Transects':
			filt = (idf.index >= min(BBp_edges)) & (idf.index <= max(BBp_edges)) & (idf['Mask'] == 1)
			# breakpoint()
		elif k_ == 'C' and f_src != 'Transects':
			filt = (idf.index >= min(CCp_edges)) & (idf.index <= max(CCp_edges)) & (idf['Mask'] == 1)
		odf = idf.copy()[filt][i_flds]
		# breakpoint()
		odf = odf.rename(columns=i_rname)
		IDF = pd.concat([IDF,odf,xdf],axis=1,ignore_index=False)
	# if k_ == 'B':
	# 	IDF.index -= BBp_offset
	# if k_ == 'C':
	# 	IDF.index -= CCp_offset
	print('Saving compiled transect to %s'%(os.path.join(OTROOT,'Compiled_%s%sp_Transect.csv'%(k_,k_))))
	IDF.to_csv(os.path.join(OTROOT,'Compiled_%s%sp_Transect.csv'%(k_,k_)),header=True,index=True)
	# Update dictionary with fully compiled 
	IDF_dict.update({k_:IDF})

# Plot Sorted & Colated Data
plt.figure(); axs = []
for i_,k_ in enumerate(['A','B','C']):
	ax = plt.subplot(3,1,i_+1)
	axs.append(ax)
	for f_ in IDF_dict[k_].filter(like='Surf').filter(like='mean').columns:
		iyr = f_[4:6]
		ax.errorbar(IDF_dict[k_].index,IDF_dict[k_][f_],yerr = IDF_dict[k_][IDF_dict[k_].filter(like='Surf').filter(like='%s_std'%(iyr)).columns[0]],capsize=3,label=iyr)
	ax.errorbar(IDF_dict[k_].index,IDF_dict[k_]['Bed18_mean'].values,yerr=IDF_dict[k_]['Bed18_std'].values,capsize=3)




#### PRE-PROCESS BOREHOLE DATA ####
print('Pre-Processing Borehole Deformation Data')
df_BHD = pd.DataFrame()
# Pull data, correct small extraction errors in index & convert to metric 
for f_ in flist_BHD:
	iS_ = pd.read_csv(f_,index_col='z(ft)')
	if 'Mean' in f_:
		oS_ = pd.Series(iS_.values.reshape(len(iS_),),name='edot_med(a-1)',index=np.round(iS_.index)*mpft)
	elif 'Lower' in f_:
		oS_ = pd.Series(iS_.values.reshape(len(iS_),),name='edot_min(a-1)',index=np.round(iS_.index)*mpft)
	elif 'Upper' in f_:
		oS_ = pd.Series(iS_.values.reshape(len(iS_),),name='edot_max(a-1)',index=np.round(iS_.index)*mpft)
	df_BHD = pd.concat([df_BHD,oS_],ignore_index=False,axis=1)

# Get gaussian representation of data
S_mean = df_BHD[['edot_min(a-1)','edot_max(a-1)']].mean(axis=1)
S_var = (df_BHD['edot_max(a-1)'] - df_BHD['edot_min(a-1)'])/(M_GAU_coef*2)**2

df_BHD = pd.concat([df_BHD,pd.DataFrame({'edot_mea(a-1)':S_mean.values,'edot_var(a-2)':S_var.values},index=df_BHD.index)],axis=1,ignore_index=False)
print('Saving pre-processed borehole data to %s'%(os.path.join(ODROOT,'Processed_Borehole_Deformation_Data.csv')))
df_BHD.to_csv(os.path.join(ODROOT,'Processed_Borehole_Deformation_Data.csv'),header=True,index=True)


##############################################
#### STEP 1: Estimate Models for Surfaces ####
##############################################
print('Starting STEP 1 - Estimating models and slopes for surfaces')
## Step 1a: Get Ice Surface Elevation & Slope Model Along Flow using Rolling WLS
S1_df = IDF_dict['A']

#######
### Process Along-Flow Transect ###
yr_list = []
df_AA_mod = pd.DataFrame()
m_a = []; Cm_a = []; yr = []; xc = []
for c_ in S1_df.filter(like='Surf').filter(like='mean').columns:
	i_yr = c_[4:6]
	iS_u = S1_df[c_]
	axs[0].plot(iS_u,label=c_)
	iS_s = S1_df['Surf%s_std'%(i_yr)]
	axs[0].fill_between(iS_s.index,iS_u.values - iS_s.values,iS_u + iS_s.values,alpha=0.5)
	idf_surf = tutil.est_slopes(iS_u,iS_s,wl=wl,ws=ws,minsamp=3,w_pow=-1,pfkw={'deg':1,'cov':'unscaled'})
	odf= pd.DataFrame({'a%s_mean'%(i_yr):-1*idf_surf['deg'].values,'a%s_var'%(i_yr):idf_surf['vdegdeg'].values,\
					   'b%s_mean'%(i_yr):idf_surf['b'].values,'b%s_var'%(i_yr):idf_surf['vbb'].values,\
					   'ab%s_var'%(i_yr):idf_surf['vdegb'].values},index=idf_surf.index)
	df_AA_mod = pd.concat([df_AA_mod,odf],axis=1,ignore_index=False)
	BBp_pick = idf_surf.iloc[np.argmin(np.abs(idf_surf.index - 4122))]
	CCp_pick = idf_surf.iloc[np.argmin(np.abs(idf_surf.index - 6200))]
	m_a.append(-1*BBp_pick['deg']); Cm_a.append(BBp_pick['vdegdeg']); yr.append(i_yr); xc.append('B')
	m_a.append(-1*CCp_pick['deg']); Cm_a.append(CCp_pick['vdegdeg']); yr.append(i_yr); xc.append('C')

print('Transects processed, extracting surface slopes nearest intersections')
df_XC_slope = pd.DataFrame({'Slope(deg)':m_a,'Slope_var(deg2)':Cm_a,'yr':yr,'CrossSection':xc})
df_XC_slope.to_csv(os.path.join(OTROOT,'Cross_Section_Extracted_Surface_Slopes.csv'),header=True,index=False)
df_AA_mod.to_csv(os.path.join(OTROOT,'AAp_Transect_Surfaces_WLS_Results.csv'),header=True,index=True)

## Step 1b: Get Ice-Surface Polynomial Models using np.polyfit routines in tutil
print("Getting bed model for B-B'")
S2_df = IDF_dict['B']
# Do bed model for B-B' section
BIDX = (S2_df['Bed18_mean'].notna()) | (S2_df['ExposedBed_mean'].notna())
S_u_bed_BBp = S2_df.filter(like='Bed').filter(like='mean')[BIDX].sum(axis=1)
S_s_bed_BBp = S2_df.filter(like='Bed').filter(like='std')[BIDX].sum(axis=1)
axs[1].errorbar(S_u_bed_BBp.index,S_u_bed_BBp.values,yerr=S_s_bed_BBp.values,label='Bed',capsize=5)
# Polynomial Fitting & Covariance Matrix for Bed
modbBBp,covbBBp = np.polyfit(S_u_bed_BBp.index,S_u_bed_BBp.values,BBpBo,cov=True)#,w=S_s_bed_BBp.values**-1,cov='unscaled')
# xvect = np.arange(-500,2010,10)
axs[1].plot(xvect,np.poly1d(modbBBp)(xvect),'k:')
# Do surface models for B-B' section
poly_BBp_dict = {'Bed':{'m':modbBBp,'Cm':covbBBp}}
for c_ in S2_df.filter(like='Surf').filter(like='mean').columns:
	i_yr = c_[4:6]
	iS_u = S2_df[c_]
	iS_s = S2_df['Surf%s_std'%(i_yr)]
	iIDX = iS_u.notna()
	# Polynomial 
	imod,icov = np.polyfit(iS_u[iIDX].index,iS_u[iIDX].values,pSo,cov=True)#w=iS_s[iIDX].values**-1,cov='unscaled')
	axs[1].plot(xvect,np.poly1d(imod)(xvect))
	poly_BBp_dict.update({i_yr:{'m':imod,'Cm':icov}})



print("Getting bed model for C-C'")
S3_df = IDF_dict['C']
# Do bed model for C-C' section
BIDX = (S3_df['Bed18_mean'].notna()) | (S3_df['ExposedBed_mean'].notna())
S_u_bed_CCp = S3_df.filter(like='Bed').filter(like='mean')[BIDX].sum(axis=1)
S_s_bed_CCp = S3_df.filter(like='Bed').filter(like='std')[BIDX].sum(axis=1)
axs[2].errorbar(S_u_bed_CCp.index,S_u_bed_CCp.values,yerr=S_s_bed_CCp.values,label='Bed',capsize=5)
# Polynomial Fitting & Covariance Matrix for Bed
modbCCp,covbCCp = np.polyfit(S_u_bed_CCp.index,S_u_bed_CCp.values,CCpBo,cov=True)#,w=S_s_bed_CCp.values**-1,cov='unscaled')
# xvect = np.arange(0,2010,10)
axs[2].plot(xvect,np.poly1d(modbCCp)(xvect),'k:')
# Do surface models for B-B' section
poly_CCp_dict = {'Bed':{'m':modbCCp,'Cm':covbCCp}}
for c_ in S2_df.filter(like='Surf').filter(like='mean').columns:
	i_yr = c_[4:6]
	iS_u = S3_df[c_]
	iS_s = S3_df['Surf%s_std'%(i_yr)]
	iIDX = iS_u.notna()
	# Polynomial 
	imod,icov = np.polyfit(iS_u[iIDX].index,iS_u[iIDX].values,pSo,cov=True)#w=iS_s[iIDX].values**-1,cov='unscaled')
	axs[2].plot(xvect,np.poly1d(imod)(xvect))
	poly_CCp_dict.update({i_yr:{'m':imod,'Cm':icov}})






#####################################################################
#### STEP 2: Invert Borehole & Geometry Values From Meier (1957) ####
#####################################################################
print('Starting STEP 2 - LOOCV Inversions for Effective Viscosity')
# Calculate Driving Stresses
Tdz = driving_stress(fWE*1000,gg,df_BHD.index.values,Sf54,a54)

oBm,oBCm = curve_fit(flowlaw,Tdz,df_BHD['edot_mea(a-1)'].values,\
			   sigma=df_BHD['edot_var(a-2)'].values**0.5)
pBm = [oBm[0]]; pBCm = [oBCm[0,0]]; LOOCV = [None];
# Do Leave One Out Cross Validation
for i_ in range(len(df_BHD)):
	loocv_ind = df_BHD.index != df_BHD.index[i_]
	iBm,iBCm = curve_fit(flowlaw,Tdz[loocv_ind],df_BHD['edot_mea(a-1)'][loocv_ind].values,\
			   	   sigma=df_BHD[loocv_ind]['edot_var(a-2)'].values**0.5)
	pBm.append(iBm[0])
	pBCm.append(iBCm[0,0])
	LOOCV.append(i_)
# Write results to DataFrame
df_B_est = pd.DataFrame({'B(Pa a1/3)':pBm,'B_var(Pa2 a2/3)':pBCm,'LOOCV':LOOCV})
# Write DataFrame to disk
df_B_est.to_csv(os.path.join(ODROOT,'Effective_Viscosity_LOOCV_results.csv'),header=True,index=False)


# n_mcmc = 20
#############################################################
#### STEP 3: Vint Estimation with MCMC Error Propagation ####
#############################################################
print('Starting STEP 3 - Vint MCMC Modeling/Error Propagation')
print("Processing B-B'")
# Process B-B' Transect
# Get Valley Transect Polynomial Models
polyB_B = poly_BBp_dict['Bed']['m']
covB_B = poly_BBp_dict['Bed']['Cm']
# Create Perturbed Bed Model
lhs_pB = mcmc.norm_lhs(polyB_B,covB_B,n_samps=n_mcmc)
print('LHS Samples Drawn')
# Create Holders for Estimated Parameters
p_idx = []; pAA = []; pPP = []; pHH = []; pSf = []; pTd = []; pBB = []; pVi = []; pyr = []; paa = []; prl = []; prr = []

# plt.figure()
# Create WLS best-fit profile
bed_disc_fit = np.poly1d(polyB_B)(xvect)
# Create holders for perturbed profile mean and variance
# bed_disc_rmean = np.zeros(len(xvect),)
bed_disc_rmean = deepcopy(bed_disc_fit)
bed_disc_var = np.zeros(len(xvect),)
# Do recursive mean and variance from LHS polynomials
for i_ in range(n_mcmc):
	xn1_u = np.poly1d(lhs_pB[i_,:])(xvect)
	xn_u = deepcopy(bed_disc_rmean)
	xn_v = deepcopy(bed_disc_var)
	bed_disc_rmean +=  (xn1_u - xn_u)/(i_ + 2)
	bed_disc_var += xn_u**2 - bed_disc_rmean**2 + ((bed_disc_rmean**2 - xn_v - xn_u**2)/(i_ + 2))
# Create holder for discretized
dict_BBp_disc = {'Bed_pmean(m asl)':bed_disc_fit,'Bed_rmean(m asl)':bed_disc_rmean,'Bed_var(m asl2)':bed_disc_var}

# Iterate Across Years
for k_ in poly_BBp_dict.keys():

	if k_ != 'Bed':
		# Get WLS optimal fit profile
		surf_disc_fit = np.poly1d(poly_BBp_dict[k_]['m'])(xvect)
		surf_disc_rmean = deepcopy(surf_disc_fit)
		surf_disc_var = np.zeros(len(xvect),)
		r_roots = np.zeros(2,)
		j_ = 1
		kyr = int(k_)
		if k_ in ['09','10','18']:
			kyr += 2000
		else:
			kyr += 1900
		print('B Transect: Processing %s'%(k_))
		# Get Perturbed Surface Models
		lhs_pS = mcmc.norm_lhs(poly_BBp_dict[k_]['m'],poly_BBp_dict[k_]['Cm'],n_samps=n_mcmc)
		# Get Cross-Section Surface Slope & Variance
		idf_slope = df_XC_slope[(df_XC_slope['CrossSection']=='B') & (df_XC_slope['yr']==k_)]
		for i_ in tqdm(range(n_mcmc)):
			# Conduct updates on recursive mean and variance profiles
			xn1_u = np.poly1d(lhs_pS[i_,:])(xvect)
			xn_u = deepcopy(surf_disc_rmean)
			xn_v = deepcopy(surf_disc_var)
			surf_disc_rmean += (xn1_u - xn_u)/(i_ + 2)
			surf_disc_var += xn_u**2 - surf_disc_rmean**2 + ((surf_disc_rmean**2 - xn_v - xn_u**2)/(i_ + 2))
			# # plot surface profile 
			# plt.plot(xvect,np.poly1d(lhs_pS[i_,:])(xvect),'.')
			# # Plot bed profile
			# plt.plot(xvect,np.poly1d(lhs_pB[i_,:])(xvect),'.')
			# Draw surface slope sample
			ialpha = np.random.normal(loc=idf_slope['Slope(deg)'].values,scale=idf_slope['Slope_var(deg2)'].values**0.5)[0]
			# Draw effective viscosity sample from inversion distribution
			iBB = np.random.normal(loc=df_B_est['B(Pa a1/3)'].mean(),scale=df_B_est['B_var(Pa2 a2/3)'].mean()**0.5) 
			# Calculate Geometric Parameters from Perturbed Data
			polyD,roots = tutil.get_diffd_polyfit_roots(lhs_pS[i_,:],lhs_pB[i_,:])
			if len(roots) == 2:
				x_roots = roots
			elif len(roots) == 3:
				x_roots = roots[1:]
			elif len(roots) > 3:
				breakpoint()
			elif len(roots) < 2:
				x_roots = roots

			if len(x_roots) == 2:
				# Get recursive mean roots
				r_roots += (x_roots - r_roots)/(j_)
				j_ += 1
				prl.append(x_roots[0])
				prr.append(x_roots[1])
				# Cross-sectional area
				iA = tutil.calc_A(polyD,x_roots,x_roots[0])
				# Ice-rock contact perimeter
				iP = tutil.estimate_arclength(polyD,10,xl=x_roots[0],xr=x_roots[1])
				# Maximum ice thickness
				iHi = tutil.estimate_Ymax(polyD,x_roots)
				# Shape factor
				iSf = tutil.calc_ShapeFactor(iHi,iA,iP)
				# Driving Stress
				iTdz = driving_stress(fWE*1000,gg,iHi,iSf,rpd*ialpha)
				# Deformation Velocity
				iVi = Vint(fWE*1000,gg,iHi,iSf,rpd*ialpha,iBB)
				pyr.append(kyr); p_idx.append(i_); 
				pAA.append(iA); pPP.append(iP); 
				pHH.append(iHi); pSf.append(iSf);
				pTd.append(iTdz); pBB.append(iBB); 
				pVi.append(iVi); paa.append(ialpha);

			else:
				pyr.append(kyr); p_idx.append(i_); 
				prl.append(np.nan); prr.append(np.nan);
				pAA.append(np.nan); pPP.append(np.nan); 
				pHH.append(np.nan); pSf.append(np.nan);
				pTd.append(np.nan); pBB.append(np.nan); 
				pVi.append(np.nan); paa.append(ialpha);
		
		# Trim recursive profles to within roots
		IND = (xvect >= (r_roots[0] - 10)) & (xvect <= (r_roots[1] + 10))
		surf_disc_fit[~IND] = np.nan
		surf_disc_rmean[~IND] = np.nan
		surf_disc_var[~IND] = np.nan
		# Save recursive profiles to save dictionary
		dict_BBp_disc.update({'Surf%s_pmean(m asl)'%(k_):surf_disc_fit,\
							  'Surf%s_rmean(m asl)'%(k_):surf_disc_rmean,\
							  'Surf%s_var (m asl2)'%(k_):surf_disc_var})


df_BBp_Vint = pd.DataFrame({'yr':pyr,'mcmc':p_idx,'A(m2)':pAA,'P(m)':pPP,'H(m)':pHH,'Sf':pSf,'root_left(m)':prl,\
				    'root_right(m)':prr,'Td(Pa)':pTd,'B(Pa a1/3)':pBB,'Vi(m a-1)':pVi,'Slp(deg)':paa,})
OFILE = os.path.join(ODROOT,'BBp_Transect_Deformation_MCMC_Results_1k.csv')
print('Saving results to %s'%(OFILE))
df_BBp_Vint.to_csv(OFILE,header=True,index=False)

df_BBp_prof = pd.DataFrame(dict_BBp_disc,index=xvect)
OFILE = os.path.join(OTROOT,'BBp_Transect_Surfaces_Poly_MCMC_Results.csv')
df_BBp_prof.to_csv(OFILE,header=True,index=True)

plt.figure()
for c_ in df_BBp_Vint['yr'].unique():
	IND = df_BBp_Vint['yr'] == c_
	plt.errorbar(c_,df_BBp_Vint['Vi(m a-1)'][IND].mean(),yerr=df_BBp_Vint['Vi(m a-1)'][IND].std(),capsize=5,fmt='k.')


print("Processing C-C'")
plt.figure()
# Process C-C' Transect
# n_mcmc = 20
# Get Valley Transect Polynomial Models
polyB_C = poly_CCp_dict['Bed']['m']
covB_C = poly_CCp_dict['Bed']['Cm']
# Create Perturbed Bed Model
lhs_pB = mcmc.norm_lhs(polyB_C,covB_C,n_samps=n_mcmc)
print('LHS Samples Drawn')
# Create Holders for Estimated Parameters
p_idx = []; pAA = []; pPP = []; pHH = []; pSf = []; pTd = []; pBB = []; pVi = []; pyr = []; paa = []; prl = []; prr = []
# plt.figure()
# Create WLS best-fit profile
bed_disc_fit = np.poly1d(polyB_C)(xvect)
# Create holders for perturbed profile mean and variance
# bed_disc_rmean = np.zeros(len(xvect),)
bed_disc_rmean = deepcopy(bed_disc_fit)
bed_disc_var = np.zeros(len(xvect),)
# Do recursive mean and variance from LHS polynomials
for i_ in range(n_mcmc):
	xn1_u = np.poly1d(lhs_pB[i_,:])(xvect)
	xn_u = deepcopy(bed_disc_rmean)
	xn_v = deepcopy(bed_disc_var)
	bed_disc_rmean +=  (xn1_u - xn_u)/(i_ + 2)
	bed_disc_var += xn_u**2 - bed_disc_rmean**2 + ((bed_disc_rmean**2 - xn_v - xn_u**2)/(i_ + 2))
# Create holder for discretized
dict_CCp_disc = {'Bed_pmean(m asl)':bed_disc_fit,'Bed_rmean(m asl)':bed_disc_rmean,'Bed_var(m asl2)':bed_disc_var}
# Iterate Across Years
for k_ in poly_CCp_dict.keys():
	if k_ != 'Bed':
		# Get WLS optimal fit profile
		surf_disc_fit = np.poly1d(poly_CCp_dict[k_]['m'])(xvect)
		# surf_disc_rmean = np.zeros(len(xvect),)
		surf_disc_rmean = deepcopy(surf_disc_fit)
		surf_disc_var = np.zeros(len(xvect),)
		r_roots = np.zeros(2,)
		j_ = 1
		kyr = int(k_)
		if k_ in ['09','10','18']:
			kyr += 2000
		else:
			kyr += 1900
		print('B Transect: Processing %s'%(k_))
		# Get Perturbed Surface Models
		lhs_pS = mcmc.norm_lhs(poly_CCp_dict[k_]['m'],poly_CCp_dict[k_]['Cm'],n_samps=n_mcmc)
		# Get Cross-Section Surface Slope & Variance
		idf_slope = df_XC_slope[(df_XC_slope['CrossSection']=='C') & (df_XC_slope['yr']==k_)]
		for i_ in tqdm(range(n_mcmc)):
			# Conduct updates on recursive mean and variance profiles
			xn1_u = np.poly1d(lhs_pS[i_,:])(xvect)
			xn_u = deepcopy(surf_disc_rmean)
			xn_v = deepcopy(surf_disc_var)
			surf_disc_rmean += (xn1_u - xn_u)/(i_ + 2)
			surf_disc_var += xn_u**2 - surf_disc_rmean**2 + ((surf_disc_rmean**2 - xn_v - xn_u**2)/(i_ + 2))
			# Draw surface slope sample
			ialpha = np.random.normal(loc=idf_slope['Slope(deg)'].values,scale=idf_slope['Slope_var(deg2)'].values**0.5)[0]
			# Draw effective viscosity sample
			iBB = np.random.normal(loc=df_B_est['B(Pa a1/3)'].mean(),scale=df_B_est['B_var(Pa2 a2/3)'].mean()**0.5) 
			# Calculate Geometric Parameters from Perturbed Data
			polyD,roots = tutil.get_diffd_polyfit_roots(lhs_pS[i_,:],lhs_pB[i_,:],rules=['real'])
			if len(roots) == 2:
				x_roots = roots
			# if k_ == '66':
			# 	breakpoint()
			elif len(roots) == 3:
				# print(roots)
				x_roots = roots[1:]
			elif len(roots) > 3:
				breakpoint()

			if len(x_roots) == 2:
				# Get recursive mean roots
				r_roots += (x_roots - r_roots)/(j_)
				j_ += 1
				prl.append(x_roots[0])
				prr.append(x_roots[1])
				# Cross-sectional area
				iA = tutil.calc_A(polyD,x_roots,x_roots[0])
				# Ice-rock contact perimeter
				iP = tutil.estimate_arclength(polyD,10,xl=x_roots[0],xr=x_roots[1])
				# Maximum ice thickness
				iHi = tutil.estimate_Ymax(polyD,x_roots)
				# Shape factor
				iSf = tutil.calc_ShapeFactor(iHi,iA,iP)
				# Driving Stress
				iTdz = driving_stress(fWE*1000,gg,iHi,iSf,rpd*ialpha)
				# Deformation Velocity
				iVi = Vint(fWE*1000,gg,iHi,iSf,rpd*ialpha,iBB)
				pyr.append(kyr); p_idx.append(i_); 
				pAA.append(iA); pPP.append(iP); 
				pHH.append(iHi); pSf.append(iSf);
				pTd.append(iTdz); pBB.append(iBB); 
				pVi.append(iVi); paa.append(ialpha);

			else:
				pyr.append(kyr); p_idx.append(i_); 
				prl.append(np.nan); prr.append(np.nan);
				pAA.append(np.nan); pPP.append(np.nan); 
				pHH.append(np.nan); pSf.append(np.nan);
				pTd.append(np.nan); pBB.append(np.nan); 
				pVi.append(np.nan); paa.append(ialpha);

		# Trim recursive profles to within roots
		IND = (xvect >= (r_roots[0] - 10)) & (xvect <= (r_roots[1] + 10))
		surf_disc_fit[~IND] = np.nan
		surf_disc_rmean[~IND] = np.nan
		surf_disc_var[~IND] = np.nan
		# Save recursive profiles to save dictionary
		dict_CCp_disc.update({'Surf%s_pmean(m asl)'%(k_):surf_disc_fit,\
							  'Surf%s_rmean(m asl)'%(k_):surf_disc_rmean,\
							  'Surf%s_var (m asl2)'%(k_):surf_disc_var})

df_CCp_Vint = pd.DataFrame({'yr':pyr,'mcmc':p_idx,'A(m2)':pAA,'P(m)':pPP,'H(m)':pHH,'Sf':pSf,'root_left(m)':prl,\
				    'root_right(m)':prr,'Td(Pa)':pTd,'B(Pa a1/3)':pBB,'Vi(m a-1)':pVi,'Slp(deg)':paa})
OFILE = os.path.join(ODROOT,'CCp_Transect_Deformation_MCMC_Results_1k.csv')
print('Saving results to %s'%(OFILE))
df_CCp_Vint.to_csv(OFILE,header=True,index=False)

df_CCp_prof = pd.DataFrame(dict_CCp_disc,index=xvect)
OFILE = os.path.join(OTROOT,'CCp_Transect_Surfaces_Poly_MCMC_Results.csv')
df_CCp_prof.to_csv(OFILE,header=True,index=True)

plt.figure()
for c_ in df_CCp_Vint['yr'].unique():
	IND = df_CCp_Vint['yr'] == c_
	plt.errorbar(c_,df_CCp_Vint['Vi(m a-1)'][IND].mean(),yerr=df_CCp_Vint['Vi(m a-1)'][IND].std(),capsize=5,fmt='k.')



##### GRAVEYARD ##### 
"""
The MCMC analysis introduces significant shift in recovered B values
throug the use of the 1955 transect 
- this is due to the use of Sf, not the LHS approach...
"""

# ##############################################################
# #### STEP 2: Invert Borehole Data for Effective Viscosity ####
# ##############################################################
# print('Starting STEP 2 - MCMC Effective Viscosity Inversion')
# B_mean = []; B_var = []; 
# # Fetch polynomial representations of the cross-section
# polyB = poly_BBp_dict['Bed']['m']
# covB = poly_BBp_dict['Bed']['Cm']
# polyS = poly_BBp_dict['55']['m']
# covS = poly_BBp_dict['55']['Cm']
# print('Generating Latin Hypercube Samples')
# # Generate LHS perturbed polynomials
# lhs_pB = mcmc.norm_lhs(polyB,covB,n_samps=n_lhs)
# print('...bed polynomials done...')
# lhs_pS = mcmc.norm_lhs(polyS,covS,n_samps=n_lhs)
# print('...surface polynomials done...')
# print('Sampling random normal strain rate values')
# # # Get perturbed surface slopes
# # lhs_alpha = np.random.normal(loc=5.61,scale=0.01,n_samps=n_lhs)
# # print('...surface slopes done...')
# # # Get perturbed strain rate data
# # rnd_edot = np.random.normal(loc=df_BHD['edot_med(a-1)'].values,\
# # 							scale=df_BHD['edot_var(a-2)'].values**0.5,\
# # 							n_samps = n_lhs)
# print('Random sample selection complete, starting perturbed inversions for B')
# # Create Holders for Results
# pHi = []; pA = []; pP = []; lhs_ind = []; Bm = []; BCm = []; palpha = [];
# # LHS processing pull loop
# for i_ in tqdm(range(n_lhs)):
# 	# Get perturbed surface slopes
# 	ialpha = np.random.normal(loc=4.41,scale=0.01)
# 	# Get perturbed strain rate data
# 	i_edotz = np.random.normal(loc=df_BHD['edot_med(a-1)'].values,\
# 								scale=df_BHD['edot_var(a-2)'].values**0.5)
# 	# Calculate Geometric Parameters from Perturbed Data
# 	polyD,roots = tutil.get_diffd_polyfit_roots(lhs_pS[i_,:],lhs_pB[i_,:])
# 	if len(roots) == 2:
# 		# Cross-sectional area
# 		iA = tutil.calc_A(polyD,roots,roots[0])
# 		# Ice-rock contact perimeter
# 		iP = tutil.estimate_arclength(polyD,10)
# 		# Maximum ice thickness
# 		iHi = tutil.estimate_Ymax(polyD,roots[:2])
# 		# Shape factor
# 		iSf = tutil.calc_ShapeFactor(iHi,iA,iP)
# 		# Driving Stress profile
# 		iTdz = driving_stress(fWE*1000,gg,df_BHD.index.values,iSf,rpd*ialpha)
# 		# Do inversion
# 		iBm,iBCm = curve_fit(flowlaw,iTdz,i_edotz)
# 		pHi.append(iHi); pA.append(iA); pP.append(iP)
# 		lhs_ind.append(i_); Bm.append(iBm[0]); BCm.append(iBCm[0,0]); palpha.append(ialpha)
# 	else:
# 		pHi.append(np.nan); pA.append(np.nan); pP.append(np.nan)
# 		lhs_ind.append(i_); Bm.append(np.nan); BCm.append(np.nan); palpha.append(np.nan)

# # Merge into dataframe and save to disk
# df_B_mcmc = pd.DataFrame({'Bm(Pa a1/3)':Bm,'Bvar(Pa2 a2/3)':BCm,\
# 						  'A(m2)':pA,'P(m)':pP,'Hi(m)':pHi,'a(deg)':palpha},\
# 						  index=lhs_ind)
# df_B_mcmc.to_csv(os.path.join(OROOT,'mcmc_B_est_%diter.csv'%(n_lhs)),header=True,index=True)

"""
NOTE: Well, the above didn't quite work, so let's go back to the modeling of B from the
      data presented in Meier (1957) that should result in a B = 0.18+/-0.01 MPa a**1/3
      estimate. Then we go ahead and do the modeling of Vint using the MCMC approach
      for everything nelse.
"""
##& merge into composite dataframes for each transect


#### STEP 1: Estimate Models for Surfaces ####
## Step 1a: Get Ice Surface Elevation & Slope Model Along Flow using Rolling WLS
"""
:: INPUTS ::

:: OUTPUTS ::

"""
## Step 1b: Get Ice-Surface Polynomial Models using np.polyfit routines in tutil
"""
:: INPUTS ::

:: OUTPUTS ::
"""
## Step 1c: Get Ice-Bed Interface Polynomial Model using np.polyfit routines in tutil


#### STEP 2: Invert Borehole Data for Effective Viscosity ####

#### STEP 3: Conduct LHS MCMC Analysis For Internal Deformation Velocities ####

