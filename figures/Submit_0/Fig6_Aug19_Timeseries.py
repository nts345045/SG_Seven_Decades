import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from SG_Seven_Decades.util.DF_util import reduce_series, rolling_cleanup
from SG_Seven_Decades.util.equations import RES_RMSD
# sys.path.append('/home/nates/ActiveProjects/SGGS/ANALYSIS/JUPYTER_NOTEBOOKS')
# from support.rolling_functions import rolling_cleanup

"""
This script serves 2 purposes:
1) Create the time-series summary figure for 2019 dynamics data from Saskatchewan Glacier
2) Create time-series for cross-correlation that have had 24hr rolling average values reduced
"""

issave = False


### Data & Metadata Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input GPS Data File (csv format)
IGPSD = os.path.join(DROOT,'processed_data','gps','rov1','S4_rov1_vels.csv')
# Input \dot{R}_{surf} Data File (csv format)
IRDSD = os.path.join(DROOT,'processed_data','runoff','S2_rdot_components.csv')


# IMPORT VELOCITIES
# df_VEL = pd.read_csv(ROOT+'/GPS/VEL_180min_ROV2.csv',parse_dates=True,index_col=0)
df_VEL = pd.read_csv(IGPSD,parse_dates=True,index_col=0)

# IMPORT IMPORT RUNOFF MODEL
df_RUN = pd.read_csv(IRDSD,parse_dates=True,index_col='Time')



##### PROCESSING SECTION ######

# DT_LOCAL = pd.Timedelta(-6,unit='hours') # This is Mountain Daylight Time, further from solar noon - catch by RBA
DT_LOCAL = pd.Timedelta(-7,unit='hours') # This is Mountain Standard Time, closer to solar noon

##### Process Velocities
sec2year = 365.24*24*3600
S_VH = (((df_VEL[['mX','mY']]**2).sum(axis=1))**0.5)*(df_VEL['mX']/df_VEL['mX'])*sec2year
S_VH.name ='VH(m/yr)'
S_VH.index = S_VH.index + DT_LOCAL
S_VH24 = rolling_cleanup(S_VH.resample('20s').mean().rolling('24H').mean(),S_VH,24,'H')
S_VH24 = S_VH24[S_VH24.index.isin(S_VH[S_VH.notna()].index)].resample('20s').mean()

VH_RES = S_VH - S_VH24

# VH4XC = reduce_series(S_VH,S_VH24)

# Calculate the rolling standard deviation from the residual (VH4XC) 
VHR_STD_G = VH_RES.std()
VHR_RMSD_G = RES_RMSD(VH_RES.values)
VHR_STD24 = VH_RES.rolling('24H').std()
VHR_STD24 = rolling_cleanup(VHR_STD24,VH_RES,24,'hour')
VHR_RMSD24 = VH_RES.rolling('24H').apply(RES_RMSD)
VHR_RMSD24 = rolling_cleanup(VHR_RMSD24,VH_RES,24,'hour')
# Normalize
VHR_STD24N = VHR_STD24/VHR_STD_G
VHR_RMSD24N = VHR_RMSD24/VHR_RMSD_G
# Enforce Data Gaps
VHR_STD24N = VHR_STD24N*(VH_RES/VH_RES)
VHR_RMSD24N = VHR_RMSD24N*(VH_RES/VH_RES)
print("Horizontal Velocities Processed")




##### Process Runoff Model
S_RO = df_RUN[['QSu(mm/hr)','Rdot(mm/hr)']].sum(axis=1)*(df_RUN['QSu(mm/hr)']/df_RUN['QSu(mm/hr)'])
S_RO.name='RO(mm/hr)'
S_RO.index = S_RO.index + DT_LOCAL
S_RO24 = rolling_cleanup(S_RO.resample('60T').mean().rolling('24H').mean(),S_RO,24,'hour')
S_RO24 = S_RO24[S_RO24.index.isin(S_RO[S_RO.notna()].index)].resample('60T').mean()

# RO4XC = reduce_series(S_RO,S_RO24)
RO_RES = S_RO - S_RO24

# Calculate the rolling standard deviation from the residual (VH4XC) 
ROR_STD_G = RO_RES.std()
ROR_RMSD_G = RES_RMSD(RO_RES.values)
ROR_STD24 = RO_RES.rolling('24H').std()
ROR_STD24 = rolling_cleanup(ROR_STD24,RO_RES,24,'hour')
ROR_RMSD24 = RO_RES.rolling('24H').apply(RES_RMSD)
ROR_RMSD24 = rolling_cleanup(ROR_RMSD24,RO_RES,24,'hour')
# Normalize
ROR_STD24N = ROR_STD24/ROR_STD_G
ROR_RMSD24N = ROR_RMSD24/ROR_RMSD_G
# Enforce Data Gaps
ROR_STD24N = ROR_STD24N*(RO_RES/RO_RES)
ROR_RMSD24N = ROR_RMSD24N*(RO_RES/RO_RES)
print("Runoff Model Processed")
# ##### REDUCED DATA COMPILATION AND RESAMPLING #####
# df_T0 = pd.DataFrame(index=[pd.Timestamp('2019-08-01T00:00:00')])
# df_XCD = pd.concat([df_T0,VH4XC,RO4XC],axis=1,ignore_index=False)
# df_XCD = df_XCD.resample('1T').mean()
# df_XCD = df_XCD.interpolate(limit_area='inside',limit_direction='both',limit=60)


#### SAVE SECTION ####
# df_XCD.to_csv(ROOT+'/Cross_Correlation_Data_24hr_Reduced_T1T.csv',index=True,header=True)

### FONT & FORMATTING CONTROL SECTION ###

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


##### FIGURE GENERATION SECTION #####
# Format Subplot Indices
# fig,axs[I_] = plt.subplots(nrows=4,ncols=1,sharex='col',figsize=(9.35,11.))
fig = plt.figure(figsize=(11.45,7.6))
# Subplot Label Plotting Parameters
lblX = pd.Timestamp('2019-08-07T00:00:00')
Yoff = 0.85
LBL = ['A','B','C']#,'D']

# Set Incremental Increasing Subplot Index
NP_ = 3
k_ = 0
Y_ = 0.9
H_ = 0.8/NP_

def yaxis_only(ax,unusedy='right',includeX=False):
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(includeX)
	ax.xaxis.set_visible(includeX)
	if unusedy == 'left':
		ax.yaxis.set_ticks_position('right')
		ax.yaxis.set_label_position('right')
	if unusedy == 'right':
		ax.yaxis.set_ticks_position('left')
		ax.yaxis.set_label_position('left')
	ax.spines[unusedy].set_visible(False)

# Xlims = [S_VH.index.min(),S_VH.index.max()]
Xlims = [pd.Timestamp('2019-08-01'),S_RO.index.max()]

axs = []

# Natu

for k_ in range(NP_):
	axs.append(fig.add_axes([0.1,Y_-(k_+1)*H_,0.8,H_]))

k_ += 1



### RUNOFF SUPPLY RATE ###
I_ = 0
# axs[I_] = fig.add_axes([0.1,Y_-(k_+1)*H_,0.8,H_])
axs[I_].plot(S_RO,'-',color='navy',label='$RS$')
axs[I_].plot(S_RO24,"-",color='skyblue',label='$\\bar{RS}$')
# axs[I_].set_ylabel('Runoff Supply\n [RS] (mm WE/hr)',color='b')
axs[I_].set_ylabel('$\\dot{R}_{surf}$\n(mm WE/hr)',rotation= 180 + 90*(-1**I_),labelpad=15*(I_%2))
# axs[I_].set_ylim(0,6)
# Window Dressing
# axs[I_].grid(linestyle=':',axis='x')
axs[I_].set_ylim(0,5)
axs[I_].yaxis.set_ticks(np.arange(0,5.1,1))
Ylims = axs[I_].get_ylim()
# axs[I_].text(lblX,np.sum(Ylims*np.array([1-Yoff,Yoff])),LBL[k_],fontweight='extra bold')
axs[I_].set_xlim(Xlims)
if int(I_) % 2 == 1:
	side = 'left'
else:
	side = 'right'
yaxis_only(axs[I_],unusedy=side,includeX=False)
# k_ += 1


### VELOCITY PLOT ###
I_ = 1
# AX = plt.subplot(411)
# axs[I_] = fig.add_axes([0.1,Y_-(k_+1)*H_,0.8,H_])
axs[I_].plot(S_VH,'-',color='maroon',label='$V_{SURF}$')
axs[I_].plot(S_VH24,"-",color='salmon',label='$\\bar{V}_{SURF}$')
# axs[I_].plot(Xlims,np.ones(2)*)
axs[I_].fill_between([S_VH.index.min(),S_VH.index.max()],[ud_min,ud_min],[ud_max,ud_max],\
				 color='k',alpha=0.25,label='$\\hat{\\nu}_{H}$')
# axs[I_][k_].set_ylabel('Velocity [m/yr]')
# axs[I_].set_ylabel('Horizontal\nVelocity\n[$\\nu_{s,H}$] (m/yr)')#,rotation='horizontal')
axs[I_].set_ylabel('$V_{surf}$ (m/yr)',rotation= 180 + 90*(-1**I_),labelpad=15*(I_%2))
# axs[I_].grid(linestyle=':',axis='x')
axs[I_].set_ylim(0,250)
axs[I_].yaxis.set_ticks(np.arange(0,300,50))
Ylims = axs[I_].get_ylim()
# axs[I_].text(lblX,np.sum(Ylims*np.array([1-Yoff,Yoff])),LBL[k_],fontweight='extra bold')
axs[I_].set_xlim(Xlims)
if int(I_) % 2 == 1:
	side = 'left'
else:
	side = 'right'
yaxis_only(axs[I_],unusedy=side,includeX=False)

# ### VERTICAL DISPLACEMENT RESIDUALS ###
# I_ = 2
# # plt.subplot(412,sharex=AX)
# # axs[I_] = fig.add_axes([0.1,Y_-(k_+1)*H_,0.8,H_])
# axs[I_].plot(S_dUZ24*100,"k-",label='$\\delta \\bar{U}_{Z}$',alpha=0.8)
# axs[I_].plot(S_dUZ*100,'k-',label='$\\delta u_{s,Vert}$',alpha=0.7)

# # axs[I_].set_ylabel('Reduced Vertical\nDisplacement\n[$\\delta u_{S,Z}$] (cm)')#,rotation='horizontal')
# # axs[I_].set_ylabel('$\\delta U_{Z}$ (cm)',rotation=270,labelpad=15)#,rotation='horizontal')
# axs[I_].set_ylabel('$\\delta U_{Z}$ (cm)',rotation= 180 + 90*(-1**I_),labelpad=15*(I_%2))
# # axs[I_][k_].set_ylabel('Detrended Vertical\nDisplacement [m]')
# # axs[I_][k_].legend()
# # axs[I_].grid(linestyle=':',axis='x')
# axs[I_].set_ylim(-10,10)
# axs[I_].yaxis.set_ticks(np.arange(-10,15,5))
# Ylims = axs[I_].get_ylim()
# # axs[I_].text(lblX,np.sum(Ylims*np.array([1-Yoff,Yoff])),LBL[k_],fontweight='extra bold')
# axs[I_].set_xlim(Xlims)
# if int(I_) % 2 == 1:
# 	side = 'left'
# else:
# 	side = 'right'
# yaxis_only(axs[I_],unusedy=side,includeX=False)


### SCALED 24-HOUR RMSD PLOTS ###
I_ = 2
# axs[I_].plot(ROR_STD24N,label='$\sigma_{R_{surf}}$')
axs[I_].plot(ROR_RMSD24N,color='blue',label='$\\dot{R}_{surf}}$')
# axs[I_].plot(VHR_STD24N,label='$\sigma_{V_{surf}}$')
axs[I_].plot(VHR_RMSD24N,color='red',label='$V_{surf}$')
# axs[I_].plot(dUZR_RMSD24N,color='gray',label='$\\delta\hat{U}_z - \\delta\hat{\\overline{U}}_z$}')
axs[I_].legend()
axs[I_].set_xlim(Xlims)
if int(I_) % 2 == 1:
	side = 'left'
else:
	side = 'right'
yaxis_only(axs[I_],unusedy=side,includeX=False)
axs[I_].set_ylabel('NRMSD')
axs[I_].yaxis.set_ticks(np.arange(0,2,0.25))
axs[I_].set_ylim([0.25,1.75])



### OVERLAY CONTINUOUS VERTICAL GRID DATELINES ###
Zpad = 0.01
axB = fig.add_axes([0.1,Y_-(k_)*H_-Zpad,0.8,0.8])

# Work-around for plotting the grid
for i_ in range(30):
	dateline = pd.Timestamp('2019-08-01T00:00:00')+i_*pd.Timedelta(1,unit='days')
	axB.plot([dateline,dateline],[Zpad/(0.8+Zpad),1],':',color='gray',linewidth=1)

## Appropriately scale date raster overlay
axB.set_xlim(Xlims)
axB.set_ylim(0,1)
# Make sure the overlay is a transparency
axB.patch.set_alpha(0)
# Hack of ticks
for SL_ in ['top','bottom','right','left']:
	axB.spines[SL_].set_visible(False)
# Keep xlabels visible
axB.xaxis.set_visible(True)
axB.set_xlabel('Local Time (UTC -7)')
axB.yaxis.set_visible(False)
for i_,iLB in enumerate(LBL):
	if i_ % 2 == 1:
		axB.text(Xlims[1]+pd.Timedelta(30,unit='hours'),0.99-.335*i_,iLB,fontweight='extra bold')
	else:
		axB.text(Xlims[0]-pd.Timedelta(40,unit='hours'),0.99-.335*i_,iLB,fontweight='extra bold')


for J_ in range(len(axs)):
	axs[J_].grid(axis='y')

### RENDER ###
plt.show()


#### SAVE TO FILE ####
if issave:
	OUT_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/v12_render'
	DPI = 200
	plt.savefig('%s/Figure_6_2019_timeseries_UTC-7_%ddpi.png'%(OUT_DIR,DPI),dpi=DPI,pad_inches=0.05,format='png')