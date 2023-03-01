import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Fig5_SevenDecades.py - This script generates Figure 5 and Tables 1-3
for the re-submission of Stevens and others (in review 2022).

Table generations is for convenience and to ensure that Fig. 5 and
Tabs. 2-3 consistently use the same MCMC results when estimating
parameter uncertainties

"""
def percent_loc(ax,xpct,ypct):
	xlims = ax.get_xlim()
	ylims = ax.get_ylim()
	xc = xlims[0] + (xlims[1] - xlims[0])*xpct
	yc = ylims[0] + (ylims[1] - ylims[0])*ypct
	return xc,yc

def dated_boxplot(ax,x_loc,x_w,data,quants=[0.1,0.25,0.5,0.75,0.9],color='k',capsize=5,label=None):
	"""
	Work-around for rendering box & whisker plots with custom x-axes.
	This produces a single boxplot-like rendering on a specified figure axis.

	Note: uses nan-compliant matplotlib statistical methods.

	:: INPUTS ::
	:type ax: matplotlib.axes object
	:param ax: Plot axis to render boxplot on
	:type x_loc: various
	:param x_loc: valid x-coordinate for center of boxplot
	:type x_w: x_loc-like
	:param x_w: width of the box for the boxplot in axis-valid units
	:type data: array-like
	:param data: input data, should be 1-D
	:type quants: array-like, MUST BE 5 VALUES
	:param quants: Quantiles to assess (per np.nanquantile)
				   Recommended spreads:
				   		quants = [0.1,0.25,0.5,0.75,0.9] (default) - (Non Gaussian)
				   		quants = [0.07,0.34,0.5,0.66,0.93] mean, +/-1sigma, and +/-2sigma (Gaussian)
	:type color: str
	:param color: valid argument for 'color' kwarg in matplotlib functions
	:type capsize: float
	:param capsize: size of caps (see matplotlib.pyplot.errorbar)
	:type label: str or None
	:param label: label to assign to assembled output handle

	:: RETURN ::
	:rtype compound_handle: tuple
	:return compound_handle: valid handle object to pass to matplotlib.pyplot.legend(handle=*)
	:rtype QV: numpy.ndarray
	:return QV: Quantile values ordered in p10, p25, p50, p75, p90

	"""
	QV = np.nanquantile(data,quants)
	CAPS = np.array([[QV[0]],[QV[-1]]])
	if Q0 > Q1:
		breakpoint()
	# yer = np.array([[qd[quantiles[0]]],[qd[quantiles[3]]]])
	h_ = ax.errorbar(x_loc,QV[2],yerr=CAPS,capsize=capsize,label=label,color=color)
	b_ = ax.fill_between([x_loc - x_w/2, x_loc + x_w/2],np.ones(2,)*QV[1],np.ones(2,)*QV[3],color=color)
	u_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[3],linewidth=1,color='white')
	l_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[1],linewidth=1,color='white')
	m_, = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[2],linewidth=2,color='white')
	return (h_,b_,u_,l_,m_), QV

def get_labeled_stats_dict(data,tag,quants):
	odict = {}
	# Do Gaussian Stats
	odict.update({'%s_mean'%(tag):np.nanmean(data)})
	odict.update({'%s_std'%(tag):np.nanstd(data)})
	# Do Quantile Spread
	quants = np.nanquantile(data,quants)
	for i_,q_ in enumerate(quants):
		odict.update({'%s_p%d'%(tag,int(quants[i_]*100)):q_})
	return



#################################################################################
#### SET DATA PATHS #############################################################
#################################################################################
# Project Directory
ROOT = os.path.join('..','..','..','..')

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

## CLIMATE DATA
# Climate Data Directory
CROOT = os.path.join(ROOT,'data','MINDSatUW','CLIMATE')
# Climate Normals for Nordegg, AB Climate Reference Station
NORMN = os.path.join(CROOT,'Temperature','Nordegg_Climate_Normals.csv')
# Homogenized Monthly Temperatures for Nordegg, AB Climate Reference Station
HOMTN = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'NORDEGG','NORDEGG_mean_monthly_stack.csv')




#################################################################################
#### LOAD DATA ##################################################################
#################################################################################

### Weather & Climate Data ###
# Load Homogenized Temperatures for Nordegg Climate Reference Station
df_WXNE = pd.read_csv(HOMTN,parse_dates=True,index_col=0)
# Load Climate Norms for Nordegg Climate Reference Station
df_NORM = pd.read_csv(NORMN,index_col=0)

### Internal Deformation and Surface Velocities & Parameters ###


# Load Upper Sector Deformation MCMC DataTable
df_USD = pd.read_csv(VUSN)
# Load Lower Sector Deformation MCMC DataTable
df_LSD = pd.read_csv(VLSN)
# Load Dates of DEMs from TM13
df_TM13 = pd.read_csv(MDEM,parse_dates=['Date'],index_col=0)

# Load Prior Published Surface Velocity Estiamtes
df_Tab1_LS = pd.read_csv(VLSP)
# df_Tab1_US = pd.read_csv()
df_Tab1_US = pd.read_csv(VUSP)



#################################################################################
##### TABLE GENERATION SECTION ##################################################
#################################################################################
# Set of Quantiles at which to characterize parameter distributions (plus nanmean & nanstd)
quants = [0.07,0.10,0.25,0.34,0.5,0.66,0.75,0.90,0.93] 

### Table 1 Generation - Prior Velocity Estimates ###

# Do some date conversions for plotting and output table
uDate = []; iDate = []; fDate = []; DT = []
for i_ in range(len(df_Tab1_US)):
	iS_ = df_Tab1_US.iloc[i_,:]
	t1 = iS_[['Month1','Day1','Year1']].values
	if np.isfinite(t1[1]):
		T1 = pd.Timestamp('%s %d %d'%(t1[0],t1[1],t1[2]))
	else:
		T1 = pd.Timestamp('%s %d'%(t1[0],t1[2]))
	t2 = iS_[['Month2','Day2','Year2']].values
	if np.isfinite(t2[1]):
		T2 = pd.Timestamp('%s %d %d'%(t2[0],t2[1],t2[2]))
	else:
		T2 = pd.Timestamp('%s %d'%(t2[0],t2[2]))
	dT = (T2 - T1).total_seconds()/(3600*24)
	Tu = T1 + pd.Timedelta(dT,unit='day')/2
	uDate.append(Tu)
	iDate.append(T1)
	fDate.append(T2)
	DT.append(dT)
df_Tab1_US = pd.concat([df_Tab1_US,pd.DataFrame({'T1':iDate,'T2':fDate,'Tu':uDate,'DT(days)':DT})],axis=1,ignore_index=False)

uDate = []; iDate = []; fDate = []; DT = []
for i_ in range(len(df_Tab1_LS)):
	iS_ = df_Tab1_LS.iloc[i_,:]
	t1 = iS_[['Month1','Day1','Year1']].values
	if np.isfinite(t1[1]):
		T1 = pd.Timestamp('%s %d %d'%(t1[0],t1[1],t1[2]))
	else:
		T1 = pd.Timestamp('%s %d'%(t1[0],t1[2]))
	t2 = iS_[['Month2','Day2','Year2']].values
	if np.isfinite(t2[1]):
		T2 = pd.Timestamp('%s %d %d'%(t2[0],t2[1],t2[2]))
	else:
		T2 = pd.Timestamp('%s %d'%(t2[0],t2[2]))
	dT = (T2 - T1).total_seconds()/(3600*24)
	Tu = T1 + pd.Timedelta(dT,unit='day')/2
	uDate.append(Tu)
	iDate.append(T1)
	fDate.append(T2)
	DT.append(dT)
df_Tab1_LS = pd.concat([df_Tab1_LS,pd.DataFrame({'T1':iDate,'T2':fDate,'Tu':uDate,'DT(days)':DT})],axis=1,ignore_index=False)


### Table 2 Generation - Transect Parameters & Physical Parameter Estimates ###
# Fetch DEM Dates
## TODO: Convert this section into a direct plotting routine

T_dep = ['yr','A(m2)','P(m)','H(m)','Sf','root_left(m)','root_right(m)','Td(Pa)','Vi(m a-1)','Slp(deg)']
Indep = ['B(Pa a1/3)']

# Upper Sector Components
df_Tab2_US = pd.DataFrame()
for y_ in np.sort(df_USD['yr'].unique()):
	if y_ != 2010:
		idf = df_USD[df_USD['yr'] == y_][T_dep]
		if y_ == 2018:
			uyr = 2019
		for f_ in T_dep:
			isd = get_labeled_stats_dict(idf[f_])
		df_Tab2_US = pd.concat([df_Tab2_US,idf_stats],axis=1,ignore_index=False)


## Lower Sector Component
df_Tab2_LS = pd.DataFrame()
plt.subplot(212)
for y_ in np.sort(df_LSD['yr'].unique()):
	if y_ != 2010:
		idf = df_LSD[df_LSD['yr'] == y_][T_dep]
		# plt.hist(idf['Vi(m a-1)'],100,label=y_,alpha=0.2,density=True)

		idf_u = idf.median()
		idf_q = idf.quantile(p_va)

		if y_ == 2018:
			uyr = '2017 mean'
			syr = '2017 p%02d'%(p_va*100)
		else:
			uyr = '%04d mean'%(y_)
			syr = '%04d p%02d'%(y_,p_va*100)
		idf_u.name = uyr
		idf_s.name = syr
		df_Tab2_LS = pd.concat([df_Tab2_LS,idf_u,idf_s],axis=1,ignore_index=False)
# plt.legend()
plt.title('Lower Sector')


# Physical Constant Estimates
B_u = np.nanmedian(np.hstack([df_USD['B(Pa a1/3)'].values,df_LSD['B(Pa a1/3)'].values]))
B_s = np.nanquantile(np.hstack([df_USD['B(Pa a1/3)'].values,df_LSD['B(Pa a1/3)'].values]),p_va) - B_u



### Table 3 Generation - Sliding Velocity Estimates ###



#################################################################################
##### CLIMATE DATA PROCESSING SECTION ###########################################
#################################################################################





#################################################################################
##### PLOTTING SECTION ##########################################################
#################################################################################
## Set Font Sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## Generate Figure
fig = plt.figure(figsize=(7,7))
## Initialize Subplots
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)


#################################################################################
#### CLIMATE & WX ###############################################################
#################################################################################

# Do temperature anomaly compared to normals or X bounding years





#################################################################################
#### UPPER SECTOR ###############################################################
#################################################################################
ax1b = ax1.twinx()
handles = []
iDu = df_Tab2_US.filter(like='mean').T.sort_values('yr')
iDs = df_Tab2_US.filter(like=' p').T.sort_values('yr')

# iDu = df_Tab3_US
# iDs = df_Tab3_US
# h_ = ax1.errorbar(iDu['yr'].values,iDu['Vs(m a-1)'].values,yerr=sig_lvl*iDs)
f_tuples = [('Td(Pa)','$\\Delta\\tau_{d}$ (%)'),\
			('Slp(deg)','$\\Delta\\alpha$ (%)'),\
			('H(m)','$\\Delta H_{i}$ (%)'),\
			('Sf','$\\Delta S_f$ (%)')]
# Plot Percent X-axis data
for f_,l_ in f_tuples:
	fld = f_
	f_bar = iDu[fld].mean()
	h_, = ax1b.plot(iDu['yr'],100*(iDu[fld].values - f_bar)/f_bar,label=l_,alpha=0.33)
	handles.append(h_)
	ax1b.fill_between(iDu['yr'],100*(iDu[fld].values - iDs[fld].values - f_bar)/f_bar,\
								100*(iDu[fld].values + iDs[fld].values - f_bar)/f_bar,\
								alpha=0.1)

# Plot Velocity Y-axis data
for y_ in np.sort(df_USD['yr'].unique()):
	if y_ != 2010:
		idf = df_USD[df_USD['yr'] == y_][T_dep]
		h_ = dated_boxplot(ax1,pd.Timestamp('%04d-01-01'%(y_)),pd.Timedelta(365,unit='day'),\
							   idf['Vi(m a-1)'].values,label='$\\hat{V}_{int}$')	


h_ = ax1.errorbar(iDu['yr'].values,iDu['Vi(m a-1)'].values,yerr=iDs['Vi(m a-1)'].values,\
				  fmt='s',label='$\\hat{V}_{int}$ ($m$ $a^{-1}$)',capsize=5)
handles.append(h_)

ax1.set_ylabel('Velocity ($m$ $a^{-1}$)')
ax1b.set_ylabel('Change From Average\n$\\Delta$ (%)',rotation=270,labelpad=15)
ax1.legend(handles=handles,ncol=5,loc='upper center')


#################################################################################
#### LOWER SECTOR ###############################################################
#################################################################################
ax2b = ax2.twinx()
handles = []
iDu = df_Tab2_LS.filter(like='mean').T.sort_values('yr')
iDs = df_Tab2_LS.filter(like=' p').T.sort_values('yr')

f_tuples = [('Td(Pa)','$\\Delta\\tau_{d}$ (%)'),\
			('Slp(deg)','$\\Delta\\alpha$ (%)'),\
			('H(m)','$\\Delta H_{i}$ (%)'),\
			('Sf','$\\Delta S_f$ (%)')]
# Plot Percent X-axis data
for f_,l_ in f_tuples:
	fld = f_
	f_bar = iDu[fld].mean()
	h_, = ax2b.plot(iDu['yr'],100*(iDu[fld].values - f_bar)/f_bar,label=l_,alpha=0.33)
	handles.append(h_)
	ax2b.fill_between(iDu['yr'],100*(iDu[fld].values - iDs[fld].values - f_bar)/f_bar,\
								100*(iDu[fld].values + iDs[fld].values - f_bar)/f_bar,\
								alpha=0.1)
# Plot Velocity Y-axis data
h_ = ax2.errorbar(iDu['yr'].values,iDu['Vi(m a-1)'].values,yerr=iDs['Vi(m a-1)'].values,\
				  fmt='s',label='$\\hat{V}_{int}$ ($m$ $a^{-1}$)',capsize=5)
handles.append(h_)

ax2.set_ylabel('Velocity ($m$ $a^{-1}$)')
ax2b.set_ylabel('Change From Average\n$\\Delta$ (%)',rotation=270,labelpad=15)
ax2.legend(handles=handles,ncol=5,loc='lower center')


plt.show()


#################################################################################
##### FIGURE AND TABLE EXPORT SECTION ###########################################
#################################################################################