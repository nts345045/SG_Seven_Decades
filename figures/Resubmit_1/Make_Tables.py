import os
import pandas as pd
import numpy as np
"""
Make_Tables.py - This script conducts a number of data merges and statistical
assessments of surface velocity data and internal deformation velocity estimates
to infer rates of basal sliding in the lower and upper sectors.

Outputs provide inputs for generating Tables 2-4 and Figure 6 in the revised
version of Stevens and others (in review, JOG).


TODO: Do a marginal quantile exclusion before doing gaussian stats (remove big outliers)
"""



def extract_stats(S_,quants,mean=True,std=True,min=True,max=True,tag=None):
	dOUT = {}
	if not isinstance(S_,pd.Series):
		# try:
		S_ = pd.Series(S_,name=tag)
		# except:
		# 	break
	if tag is None:
		tag = S_.name
	if mean:
		S_u = S_.mean()
		dOUT.update({'%s_mean'%(tag):S_u})
	if std:
		S_s = S_.std()
		dOUT.update({'%s_std'%(tag):S_s})
	if min:
		S_m = S_.min()
		dOUT.update({'%s_min'%(tag):S_m})
	if max:
		S_M = S_.max()
		dOUT.update({'%s_min'%(tag):S_M})
	for q_ in quants:
		S_q = S_.quantile(q_)
		dOUT.update({'%s_q%03d'%(tag,int(q_*100)):S_q})
	dOUT.update({'%s_IQR'%(tag):S_.quantile(.75) - S_.quantile(.25)})
	return dOUT



#################################################################################
#### SET DATA PATHS #############################################################
#################################################################################
# Project Directory
ROOT = os.path.join('..','..','..','..')
# Output Directory
ODIR = os.path.join(ROOT,'results','Main','Resubmit_1')

## INTERNAL DEFORMATION DATA
# Deformation Model Results Directory
DROOT = os.path.join(ROOT,'processed_data','deformation')
# Surface Velocity Data
VESTS = os.path.join(ROOT,'processed_data','GPS','continuous','S4')
# Previous Surface Velocity Estimates
VUSP = os.path.join(ROOT,'data','MINDSatUW','SURFACE_VELOCITIES','Resubmit_1','Upper_Sector_Surface_Velocities.csv')
VLSP = os.path.join(ROOT,'data','MINDSatUW','SURFACE_VELOCITIES','Resubmit_1','Lower_Sector_Surface_Velocities.csv')
# MCMC Internal Deformation Velocity Estimates and Parameters
VUSN = os.path.join(DROOT,'BBp_Transect_Deformation_MCMC_Results_1k.csv')
VLSN = os.path.join(DROOT,'CCp_Transect_Deformation_MCMC_Results_1k.csv')
# DEM dates (specifically) (Extracted from Table 2 in Tennant & Menounos, 2013)
MDEM = os.path.join(ROOT,'data','MINDSatUW','GIS','Transects','TM13','Tennant_Menounos_Table2_3_merge.csv')
# ETI Melt Model Parameters
METI = os.path.join(ROOT,'processed_data','runoff','model_coefficients_and_parameters.csv')
# Table 1 Location 
PDLF = os.path.join(ROOT,'results','Main','Resubmit_1','Lower_Sector_Surface_Velocities_DEM.csv')
PDUF = os.path.join(ROOT,'results','Main','Resubmit_1','Upper_Sector_Surface_Velocities_DEM.csv')

# Load Upper Sector Deformation MCMC DataTable
df_USD = pd.read_csv(VUSN)
# Load Lower Sector Deformation MCMC DataTable
df_LSD = pd.read_csv(VLSN)
# Load Dates of DEMs from TM13
df_TM13 = pd.read_csv(MDEM,parse_dates=['Date'],index_col=0)
# Load ETI Model Results
df_ETI = pd.read_csv(METI)
# Load DEM & Surface Velocity Results
df_PDL = pd.read_csv(PDLF,parse_dates=['T1','T2'])
df_PDU = pd.read_csv(PDUF,parse_dates=['T1','T2'])




################################################
##### TABLE 1 - PAST VELOCITY MEASUREMENTS #####
################################################
# Manually Generated

############################################################
##### TABLE 2 - INVERSION RESULTS & PHYSICAL CONSTANTS #####
############################################################
quants = [0.1,0.25,0.34,0.5,0.66,0.75,0.9]
B_vect = np.hstack([df_USD['B(Pa a1/3)'].values,df_LSD['B(Pa a1/3)']])
B_stats = extract_stats(B_vect,quants,tag='B')
ETI_data = df_ETI.T
fWE = 0.91
rho_ice = fWE*1000
means = [B_stats['B_mean']/1e6,\
		 df_ETI['fI(mmWE-m2/hr-W)'].values[0]*24,\
		 df_ETI['fT(mmWE/hr-C)'].values[0]*24,\
		 rho_ice]
meds = [B_stats['B_q050']/1e6,np.nan,np.nan,np.nan]
StDevs = [B_stats['B_std']/1e6,\
		  24*df_ETI['fIvar(mmWE2-m4/hr2-W2)'].values[0]*24,\
		  24*df_ETI['fTvar(mmWE2/hr2-C2)'].values[0]**0.5,\
		  np.nan]
IQR = [B_stats['B_IQR']/2e6,np.nan,np.nan,np.nan]

df_INV_RESULTS = pd.DataFrame({'mean':means,'std':StDevs,'median':meds,'IQR/2':IQR},\
							  index=['B(MPa a1/3)','fI(mmWE m2 d-1 W-1)','fT(mmWE d-1 C-1)','pICE(kg m-3)'])

df_INV_RESULTS.to_csv(os.path.join(ODIR,'Table_2_Inversion_Parameter_Results.csv'),header=True,index=True)

##########################################
##### TABLE 3 - GEOMETRIC PARAMETERS #####
##########################################
flds = ['A(m2)','P(m)','H(m)','Td(Pa)','Slp(deg)','Sf']
means = {}
meds = {}
stds = {}
hIQRs = {}
dates = {}
for f_ in flds:
	for y_ in df_LSD['yr'].unique():
		if y_ == 2018:
			yl = 2019
		else:
			yl = y_
		if y_ != 2010:
			IND = df_LSD['yr'] == y_
			# Remove large outlier population (physically unrealistic values)
			if f_ == 'Sf':
				FILT = (df_LSD[f_][IND] >= df_LSD[f_][IND].quantile(0.001))&\
					   (df_LSD[f_][IND] <= df_LSD[f_][IND].quantile(0.99))
			else:
				FILT = (df_LSD[f_][IND] >= df_LSD[f_][IND].quantile(0.001))&\
					   (df_LSD[f_][IND] <= df_LSD[f_][IND].quantile(0.999))
			means.update({'%d_%s'%(yl,f_):df_LSD[f_][IND][FILT].mean()})
			meds.update({'%d_%s'%(yl,f_):df_LSD[f_][IND][FILT].median()})
			stds.update({'%d_%s'%(yl,f_):df_LSD[f_][IND][FILT].std()})
			hIQRs.update({'%d_%s'%(yl,f_):df_LSD[f_][IND][FILT].quantile([0.25,0.75]).diff().values[1]/2})
			dates.update({'%d_%s'%(yl,f_):df_PDL[df_PDL['Years']==str(yl)]['T1'].values[0]})


df_GLS = pd.DataFrame({'mean':means,'median':meds,'stdev':stds,'IQR/2':hIQRs,'Dates':dates})
df_GLS.to_csv(os.path.join(ODIR,'Table_3b_Lower_Sector_Geometries.csv'),header=True,index=True)

means = {}
meds = {}
stds = {}
hIQRs = {}
dates = {}
for f_ in flds:
	for y_ in df_USD['yr'].unique():
		if y_ == 2018:
			yl = 2017
		else:
			yl = y_
		if y_ != 2010:
			IND = df_USD['yr'] == y_
			FILT = (df_USD[f_][IND] >= df_USD[f_][IND].quantile(0.001))&\
				   (df_USD[f_][IND] <= df_USD[f_][IND].quantile(0.999))
			means.update({'%d_%s'%(yl,f_):df_USD[f_][IND][FILT].mean()})
			meds.update({'%d_%s'%(yl,f_):df_USD[f_][IND][FILT].median()})
			stds.update({'%d_%s'%(yl,f_):df_USD[f_][IND][FILT].std()})
			hIQRs.update({'%d_%s'%(yl,f_):df_USD[f_][IND][FILT].quantile([0.25,0.75]).diff().values[1]/2})
			# means.update({'%d_%s'%(yl,f_):df_USD[f_][df_USD['yr']==y_].mean()})
			# meds.update({'%d_%s'%(yl,f_):df_USD[f_][df_USD['yr']==y_].median()})
			# stds.update({'%d_%s'%(yl,f_):df_USD[f_][df_USD['yr']==y_].std()})
			# hIQRs.update({'%d_%s'%(yl,f_):df_USD[f_][df_USD['yr']==y_].quantile([0.25,0.75]).diff().values[1]/2})
			dates.update({'%d_%s'%(yl,f_):df_PDU[df_PDU['Years']==str(yl)]['T1'].values[0]})


df_GUS = pd.DataFrame({'mean':means,'median':meds,'stdev':stds,'IQR/2':hIQRs,'Dates':dates})
df_GUS.to_csv(os.path.join(ODIR,'Table_3a_Upper_Sector_Geometries.csv'),header=True,index=True)


###################################################
##### TABLE 4 / FIGURE 6 - VELOCITY ESTIMATES #####
###################################################
# Get compiled vector of surface velocity estiamtes

IND = (df_PDL['Tag']=='Velocity') & (df_PDL['V'].notna())
# Get mean time for velocity measurements
tVs = []
dtVs = []
for i_ in range(len(df_PDL[IND])):
	idtu = pd.Timedelta((df_PDL[IND]['T2'].iloc[i_] - df_PDL[IND]['T1'].iloc[i_]).total_seconds(),unit='sec')/2
	dtVs.append(idtu.total_seconds()/(3600*24))
	tVs.append(df_PDL[IND]['T1'].iloc[i_] + idtu)
uVs = df_PDL[IND]['V'].values
sVs = df_PDL[IND]['std'].values
season = df_PDL[IND]['Season'].values

udates = []
uVi = []
sVi = []
q1Vi = []
q3Vi = []
p03Vi = []
p05Vi = []
p34Vi = []
p66Vi = []
p95Vi = []
p97Vi = []
hIQRi = []
mVi = []

# Extract internal defromation velocity stats and specific date
for y_ in df_LSD['yr'].unique():
	if y_ != 2010:
		if y_ == 2018:
			yl = 2019
		else:
			yl = y_
		S_Vi = df_LSD[df_LSD['yr']==y_]['Vi(m a-1)']
		V_date = df_PDL[df_PDL['Years']==str(yl)]['T1'].values[0]
		udates.append(V_date)
		uVi.append(S_Vi.mean())
		sVi.append(S_Vi.std())
		mVi.append(S_Vi.median())
		q1Vi.append(S_Vi.quantile(.25))
		p03Vi.append(S_Vi.quantile(.03))
		p05Vi.append(S_Vi.quantile(.05))
		p34Vi.append(S_Vi.quantile(.34))
		p66Vi.append(S_Vi.quantile(.66))
		p95Vi.append(S_Vi.quantile(.95))
		p97Vi.append(S_Vi.quantile(.97))
		q3Vi.append(S_Vi.quantile(.75))
		hIQRi.append((S_Vi.quantile(.97) - S_Vi.quantile(.03))/2)

df_VsL = pd.DataFrame({'Vs_mean':uVs,'Vs_std':sVs,'season':season,'DT/2(days)':dtVs},index=tVs)
df_ViL = pd.DataFrame({'Vi_mean':uVi,'Vi_std':sVi,'Vi_med':mVi,'Vi_Q1':q1Vi,'Vi_Q3':q3Vi,\
					   'Vi_p03':p03Vi,'Vi_p05':p05Vi,'Vi_p34':p34Vi,'Vi_p66':p66Vi,\
					   'Vi_p95':p95Vi,'Vi_p97':p97Vi,'Vi_p03QR/2':hIQRi},index=udates)
df_VLS = pd.concat([df_VsL,df_ViL],axis=1,ignore_index=False)

df_VLS.to_csv(os.path.join(ODIR,'Lower_Sector_Compiled_Velocities_RAW.csv'),header=True,index=True)


# Interpolate and Estimate V_slip @ 2-sigma
df_VLSi = df_VLS.copy().interpolate()
IND = df_VLS['Vs_mean'].notna()
uVslip = df_VLSi[IND]['Vs_mean'] - df_VLSi[IND]['Vi_mean']
uVslip.name = 'Vslip_mean'
mVslip = df_VLSi[IND]['Vs_mean'] - df_VLSi[IND]['Vi_med']
mVslip.name = 'Vslip_med'
sVslip = df_VLSi[IND]['Vs_std'] + df_VLSi[IND]['Vi_std']
sVslip.name = 'Vslip_std'
p34Vslip = (df_VLSi[IND]['Vs_std']**2 + ((df_VLSi[IND]['Vi_p66'] - df_VLSi[IND]['Vi_p34'])/2)**2)**0.5
p34Vslip.name = 'Vslip_p66_err'
p03Vslip = ((2*df_VLSi[IND]['Vs_std'])**2 + ((df_VLSi[IND]['Vi_p97'] - df_VLSi[IND]['Vi_p03'])/2)**2)**0.5
p03Vslip.name = 'Vslip_p97_err'
p05Vslip = ((df_VLSi[IND]['Vs_std']*1.96)**2 + ((df_VLSi[IND]['Vi_p95'] - df_VLSi[IND]['Vi_p05'])/2)**2)**0.5
p05Vslip.name = 'Vslip_p95_err'
df_VLSi = pd.concat([df_VLSi,uVslip,mVslip,sVslip,p34Vslip,p05Vslip,p03Vslip],axis=1,ignore_index=False)
df_VLSi.to_csv(os.path.join(ODIR,'Lower_Sector_Basal_Sliding_Velocity_Model.csv'),header=True,index=True)



##########################################
#### UPPER SECTOR VELOCITY STATISTICS ####
##########################################

IND = (df_PDU['Tag']=='Velocity') & (df_PDU['V'].notna())
# Get mean time for velocity measurements
tVs = []; dtVs = []
for i_ in range(len(df_PDU[IND])):
	idtu = pd.Timedelta((df_PDU[IND]['T2'].iloc[i_] - df_PDU[IND]['T1'].iloc[i_]).total_seconds(),unit='sec')/2
	dtVs.append(idtu.total_seconds()/(3600*24))
	tVs.append(df_PDU[IND]['T1'].iloc[i_] + idtu)
uVs = df_PDU[IND]['V'].values
sVs = df_PDU[IND]['std'].values
season = df_PDU[IND]['Season'].values

udates = []
uVi = []
sVi = []
q1Vi = []
q3Vi = []
p03Vi = []
p05Vi = []
p34Vi = []
p66Vi = []
p95Vi = []
p97Vi = []
hIQRi = []
mVi = []

# Extract internal defromation velocity stats and specific date
for y_ in df_USD['yr'].unique():
	if y_ != 2010:
		if y_ == 2018:
			yl = 2017
		else:
			yl = y_
		S_Vi = df_USD[df_USD['yr']==y_]['Vi(m a-1)']
		V_date = df_PDU[df_PDU['Years']==str(yl)]['T1'].values[0]
		udates.append(V_date)
		uVi.append(S_Vi.mean())
		sVi.append(S_Vi.std())
		mVi.append(S_Vi.median())
		q1Vi.append(S_Vi.quantile(.25))
		p03Vi.append(S_Vi.quantile(.03))
		p05Vi.append(S_Vi.quantile(.05))
		p34Vi.append(S_Vi.quantile(.34))
		p66Vi.append(S_Vi.quantile(.66))
		p95Vi.append(S_Vi.quantile(.95))
		p97Vi.append(S_Vi.quantile(.97))
		q3Vi.append(S_Vi.quantile(.75))
		hIQRi.append((S_Vi.quantile(.97) - S_Vi.quantile(.03))/2)

df_VsU = pd.DataFrame({'Vs_mean':uVs,'Vs_std':sVs,'season':season,'DT/2(days)':dtVs},index=tVs)
df_ViU = pd.DataFrame({'Vi_mean':uVi,'Vi_std':sVi,'Vi_med':mVi,'Vi_Q1':q1Vi,'Vi_Q3':q3Vi,\
					   'Vi_p03':p03Vi,'Vi_p05':p05Vi,'Vi_p34':p34Vi,'Vi_p66':p66Vi,\
					   'Vi_p95':p95Vi,'Vi_p97':p97Vi,'Vi_p03QR/2':hIQRi},index=udates)
df_VUS = pd.concat([df_VsU,df_ViU],axis=1,ignore_index=False)

df_VUS.to_csv(os.path.join(ODIR,'Upper_Sector_Compiled_Velocities_RAW.csv'),header=True,index=True)


# Interpolate and Estimate V_slip @ 2-sigma
df_VUSi = df_VUS.copy().interpolate()
IND = df_VUS['Vs_mean'].notna()
uVslip = df_VUSi[IND]['Vs_mean'] - df_VUSi[IND]['Vi_mean']
uVslip.name = 'Vslip_mean'
mVslip = df_VUSi[IND]['Vs_mean'] - df_VUSi[IND]['Vi_med']
mVslip.name = 'Vslip_med'
sVslip = (df_VUSi[IND]['Vs_std']**2 + df_VUSi[IND]['Vi_std']**2)**0.5
sVslip.name = 'Vslip_std'
p34Vslip = (df_VUSi[IND]['Vs_std']**2 + ((df_VUSi[IND]['Vi_p66'] - df_VUSi[IND]['Vi_p34'])/2)**2)**0.5
p34Vslip.name = 'Vslip_p66_err'
p03Vslip = ((2*df_VUSi[IND]['Vs_std'])**2 + ((df_VUSi[IND]['Vi_p97'] - df_VUSi[IND]['Vi_p03'])/2)**2)**0.5
p03Vslip.name = 'Vslip_p97_err'
p05Vslip = ((1.96*df_VUSi[IND]['Vs_std'])**2 + ((df_VUSi[IND]['Vi_p95'] - df_VUSi[IND]['Vi_p05'])/2)**2)**0.5
p05Vslip.name = 'Vslip_p95_err'
df_VUSi = pd.concat([df_VUSi,uVslip,mVslip,sVslip,p34Vslip,p05Vslip,p03Vslip],axis=1,ignore_index=False)
df_VUSi.to_csv(os.path.join(ODIR,'Upper_Sector_Basal_Sliding_Velocity_Model.csv'),header=True,index=True)

