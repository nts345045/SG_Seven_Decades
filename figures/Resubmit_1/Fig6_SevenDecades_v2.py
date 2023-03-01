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

def dated_boxplot_analytic(ax,x_loc,x_w,data,quants=[0.1,0.25,0.5,0.75,0.9],color='k',capsize=5,label=None):
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
	u_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[3],linewidth=1,color='white',alpha=0.5)
	l_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[1],linewidth=1,color='white',alpha=0.5)
	m_, = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*QV[2],linewidth=1,color='white',alpha=0.25)
	return (h_,b_,u_,l_,m_), QV


def x_boxplot(ax,x_loc,x_w,mean,Qs,Caps,color='k',capsize=5,label=None):
	if len(Caps) == 2:
		Caps = np.array([[Caps[0]],[Caps[1]]])
	h_ = ax.errorbar(x_loc,mean,yerr=Caps,capsize=capsize,label=label,color=color)
	b_ = ax.fill_between([x_loc - x_w/2, x_loc + x_w/2],np.ones(2,)*Qs[0],np.ones(2,)*Qs[1],color=color)
	# u_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*Qs[1],linewidth=1,color='white')
	# l_ = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*Qs[0],linewidth=1,color='white')
	m_, = ax.plot([x_loc - x_w/2,x_loc + x_w/2],np.ones(2,)*mean,linewidth=1,color='white',alpha=0.75)
	# return (h_,b_,u_,l_,m_)
	return (h_,b_,m_)



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

## VELOCITY ESTIMATES
VUS = os.path.join(ROOT,'results','Main','Resubmit_1','Upper_Sector_Basal_Sliding_Velocity_Model.csv')
VUSR = os.path.join(ROOT,'results','Main','Resubmit_1','Upper_Sector_Compiled_Velocities_RAW.csv')
VLS = os.path.join(ROOT,'results','Main','Resubmit_1','Lower_Sector_Basal_Sliding_Velocity_Model.csv')
VLSR = os.path.join(ROOT,'results','Main','Resubmit_1','Lower_Sector_Compiled_Velocities_RAW.csv')
## Geometric Parameters
GUS = os.path.join(ROOT,'results','Main','Resubmit_1','Table_3a_Upper_Sector_Geometries.csv')
GLS = os.path.join(ROOT,'results','Main','Resubmit_1','Table_3b_Lower_Sector_Geometries.csv')

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
df_VUS = pd.read_csv(VUS,parse_dates=True,index_col=0)
df_VUSr = pd.read_csv(VUSR,parse_dates=True,index_col=0)
df_VLS = pd.read_csv(VLS,parse_dates=True,index_col=0)
df_VLSr = pd.read_csv(VLSR,parse_dates=True,index_col=0)

df_GUS = pd.read_csv(GUS,index_col=0,parse_dates=['Dates'])
df_GLS = pd.read_csv(GLS,index_col=0,parse_dates=['Dates'])



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
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222,sharex=ax0)
ax2 = fig.add_subplot(223)
sc_dict = {'Winter':'dodgerblue','Summer':'maroon','Annual':'black','Spring':'maroon','Autumn/Winter':'dodgerblue'}



#################################################################################
#### UPPER SECTOR ###############################################################
#################################################################################
flds = ['H(m)','Sf','Slp(deg)','Td(Pa)']
c_cycle = {'H(m)':'dimgrey','Sf':'maroon','Slp(deg)':'darkblue','Td(Pa)':'black'}
l_dict = {'H(m)':'H_{ice}','Sf':'S_{f}','Slp(deg)':'\\alpha','Td(Pa)':'\\tau_{d}'}


ax1b = ax1.twinx()
handles = []

# Plot Velocity Data & Models
IND = df_VUSr['Vs_mean'].notna()
sdf = df_VUSr[IND]
idf = df_VUSr[~IND]

for s_ in range(len(sdf)):
	isdf = sdf.iloc[s_]
	if 2*isdf['Vs_std'] > 1:
		x_boxplot(ax1,isdf.name,pd.Timedelta(365,unit='day'),isdf['Vs_mean'],\
				  [isdf['Vs_mean'] - isdf['Vs_std'],isdf['Vs_mean'] + isdf['Vs_std']],\
				  [2*isdf['Vs_std']],\
				  color=sc_dict[isdf['season']])
	else:
		ax1.scatter(isdf.name,isdf['Vs_mean'],c=sc_dict[isdf['season']],edgecolor='dimgrey',marker='s',facecolor=None)

for i_ in range(len(idf)):
	iidf = idf.iloc[i_]
	x_boxplot(ax1,iidf.name,pd.Timedelta(365,unit='day'),iidf['Vi_med'],\
			  [iidf['Vi_p34'],iidf['Vi_p66']],\
			  [iidf['Vi_med'] - iidf['Vi_p05'],iidf['Vi_p95'] - iidf['Vi_med']],\
			  color='purple')


for f_ in flds:
	idf = df_GUS.T.filter(like=f_).T.sort_values('Dates')
	iDu = idf['mean']
	iDs = idf['IQR/2']
	# breakpoint()
	ax1b.plot(idf['Dates'],100*(iDu - iDu.mean())/iDu.mean(),'-',color=c_cycle[f_],alpha=0.25)
	ax1b.plot(idf['Dates'],100*(iDu - iDs - iDu.mean())/iDu.mean(),':',color=c_cycle[f_],alpha=0.25)
	ax1b.plot(idf['Dates'],100*(iDu + iDs - iDu.mean())/iDu.mean(),':',color=c_cycle[f_],alpha=0.25)

	# ax1b.fill_between(idf['Dates'].values,\
	# 				  100*(iDu.values - iDs.values - iDu.mean())/iDu.mean(),\
	# 				  100*(iDu.values + iDs.values - iDu.mean())/iDu.mean(),\)


ax1.set_ylabel('Velocity ($m$ $a^{-1}$)')
ax1b.set_ylabel('Change From Average\n$\\Delta$ (%)',rotation=270,labelpad=30)
ax1.legend(handles=handles,ncol=5,loc='upper center')


#################################################################################
#### LOWER SECTOR ###############################################################
#################################################################################
ax2b = ax2.twinx()
handles = []

for f_ in flds:
	idf = df_GLS.T.filter(like=f_).T.sort_values('Dates')
	iDu = idf['mean']
	iDs = idf['IQR/2']
	# breakpoint()
	ax2b.plot(idf['Dates'],100*(iDu - iDu.mean())/iDu.mean(),'-',color=c_cycle[f_],alpha=0.33,\
		label='$\\Delta %s$ (%s)'%(l_dict[f_],'%'))
	ax2b.plot(idf['Dates'],100*(iDu - iDs - iDu.mean())/iDu.mean(),':',color=c_cycle[f_],alpha=0.33)
	ax2b.plot(idf['Dates'],100*(iDu + iDs - iDu.mean())/iDu.mean(),':',color=c_cycle[f_],alpha=0.33)



# Plot Velocity Data & Models
IND = df_VLSr['Vs_mean'].notna() 
sdf = df_VLSr[IND & (df_VLSr.index.notna())]
idf = df_VLSr[~IND & (df_VLSr.index.notna())]

for s_ in range(len(sdf)):
	isdf = sdf.iloc[s_]
	if 2*isdf['Vs_std'] > 1:
		x_boxplot(ax2,isdf.name,pd.Timedelta(365,unit='day'),isdf['Vs_mean'],\
				  [isdf['Vs_mean'] - isdf['Vs_std'],isdf['Vs_mean'] + isdf['Vs_std']],\
				  [2*isdf['Vs_std']],\
				  color=sc_dict[isdf['season']])
	else:
		ax2.scatter(isdf.name,isdf['Vs_mean'],c=sc_dict[isdf['season']],edgecolor='dimgrey',marker='s',facecolor=None)

for i_ in range(len(idf)):
	iidf = idf.iloc[i_]
	x_boxplot(ax2,iidf.name,pd.Timedelta(365,unit='day'),iidf['Vi_med'],\
			  [iidf['Vi_p34'],iidf['Vi_p66']],\
			  [iidf['Vi_med'] - iidf['Vi_p05'],iidf['Vi_p95'] - iidf['Vi_med']],\
			  color='purple')



ax2b.set_ylim([-50,50])
ax2.set_ylabel('Velocity ($m$ $a^{-1}$)')
ax2b.set_ylabel('Change From Average\n$\\Delta$ (%)',rotation=270,labelpad=30)
ax2.legend(handles=handles,ncol=5,loc='upper center')
ax2b.legend(ncol=4,loc='lower left')
plt.show()


