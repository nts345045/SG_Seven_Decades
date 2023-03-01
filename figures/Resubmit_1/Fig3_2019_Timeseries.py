import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
### TO DO  ####
- Get T & I Uncertainties
- Get rainfall rate uncertainties
- Sanity check melt rate values.

"""
# Common reference location (root directory)
ROOT = os.path.join('..','..','..','..')
# Output Path
OUTD = os.path.join(ROOT,'results','Main','Resubmit_1')
# Enhanced Temperature Index Model
ETI = os.path.join(ROOT,'processed_data','runoff','ETI_Surface_Runoff_Rates_Resampled.csv')
# Surface Velocity Data
VSURF = os.path.join(ROOT,'processed_data','gps','continuous','S4C_rov1_vels.csv')
# Internal Deformation Velocity Model
VINT = os.path.join(ROOT,'processed_data','deformation','CCp_Transect_Deformation_MCMC_Results_1k.csv')
# Output Root
ODIR = os.path.join(ROOT,'results','Main','Resubmit_1')
issave = True
DPI = 200
def yaxis_only(ax,unusedy='right',includeX=False):
	"""
	Wrapper for variably displaying already-rendered X and Y axes
	"""
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

def plot_error_bounds(ax,S_med,S_var,sig_lvl=2,color='dodgerblue',alpha=0.25,label=None):
	lines = ax.plot(S_med,color=color,label=label)
	polys = ax.fill_between(S_med.index,S_med.values - sig_lvl*S_var.values**0.5,\
								S_med.values + sig_lvl*S_var.values**0.5,
								alpha=alpha,color=color)
	return lines,polys

### Unit Conversion Factors ###
spa = 3600*24*365.24 # [sec a-1] Seconds per average orbital period of Earth
sig_lvl = 2
DT_LOCAL = pd.Timedelta(7,unit='hour')

### LOAD DATA ###
df_ETI = pd.read_csv(ETI,parse_dates=True,index_col=0).interpolate(limit=1,limit_direction='both')
df_ETI.index -= DT_LOCAL
df_VS = pd.read_csv(VSURF,parse_dates=True,index_col=0)
df_VS.index -= DT_LOCAL
df_VI = pd.read_csv(VINT)

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


### PLOTTING SECTION ####

fig = plt.figure(figsize=(11.45,7.6))
# Subplot Label Plotting Parameters
lblX = pd.Timestamp('2019-08-07T00:00:00')
Yoff = 0.85
LBL = ['a','b','c'] # Subplot Labels

# Set Incremental Increasing Subplot Index
NP_ = 3
k_ = 0
Y_ = 0.9
H_ = 0.8/NP_

# Specify Uniform X-axis limits for local date-times
Xlims = [pd.Timestamp('2019-08-01T05:00'),pd.Timestamp('2019-08-22T05:00')]

# Geneate Plot Axes
axs = []
for k_ in range(NP_):
	axs.append(fig.add_axes([0.1,Y_-(k_+1)*H_,0.8,H_]))

### PLOT AIR TEMPERATURES, SOLAR INSOLATION ###
I_ = 0
if I_ == 0:
	h0, = axs[I_].plot(df_ETI['TEMP(C)'],color='black',label='$T$')
	axs[I_].set_ylabel('Temperature\n$T$ ($\\degree C$)',color='black')
	axs[I_].set_ylim([-2,22])
	axs[I_].xaxis.set_visible(False)
	axs[I_].spines['top'].set_visible(False)
	axs[I_].spines['bottom'].set_visible(False)
	axs[I_].set_xlim(Xlims)
	axs[I_].grid(axis='x',color='gray',linestyle=':')
	# Create Second Y-Axis
	axa2 = axs[I_].twinx()
	h1, = axa2.plot(df_ETI['SFLUX(w/m2)'],color='blue',label='$I$')
	axa2.set_ylabel('Incoming Shortwave\n$I$ ($W$ $m^{-2}$)',rotation=270,labelpad=30)
	axa2.set_ylim([-100,1100])
	axa2.xaxis.set_visible(False)
	axa2.spines['top'].set_visible(False)
	axa2.spines['bottom'].set_visible(False)
	axa2.set_xlim(Xlims)
	handlesA = [h0,h1]
	plt.legend(handles=handlesA,loc='upper left',ncol=2)
	y_lims = axs[I_].get_ylim()
	for i_ in range(30):
		dateline = pd.Timestamp('2019-08-01T00:00:00')+i_*pd.Timedelta(1,unit='days')
		axs[I_].plot([dateline,dateline],[y_lims[0]-50,y_lims[1]*1.3],':',color='gray',linewidth=1)

### PLOT SURFACE RUNOFF ELEMENTS ###
I_ = 1
if I_ == 1:
	# Process & Plot Cumulative Daily Runoff
	t0 = pd.Timestamp('2019-08-01T04:00')
	tf = t0 + pd.Timedelta(1,unit='day')
	i_ = 0
	axb2 = axs[I_].twinx()
	while t0 < df_ETI.index.max():
		idf = df_ETI[(df_ETI.index >= t0) & (df_ETI.index < tf)]
		nnan = np.sum(idf['TEMP(C)'].isna())
		nfin = np.sum(idf['TEMP(C)'].notna())
		S_gap = (idf['RdotM(mmWE/hr)']/idf['RdotM(mmWE/hr)'])
		RdotMi = idf['RdotM(mmWE/hr)']*S_gap
		RMi = np.nansum(RdotMi.values)*(len(idf)/nfin)
		RdotRi = idf['RdotR(mm/hr)']*S_gap
		RRi = np.nansum(RdotRi.values)*(len(idf)/nfin)
		RdotSi = RdotMi + RdotRi
		RSi = RMi + RRi
		# If no data gaps, plot full color density box
		if nnan > 0 or t0 < df_ETI.index.min() or tf > df_ETI.index.max():
			h3b = axb2.fill_between([t0,tf],[0,0],[RSi,RSi],edgecolor='cyan',facecolor='cyan',alpha=0.25,label='Partial Day')
		# If data gaps are present, partial color density
		else:
			h3a = axb2.fill_between([t0,tf],[0,0],[RSi,RSi],edgecolor='blue',facecolor='blue',alpha=0.25,label='Complete Day')
		# Advance indicies
		i_ += 1
		t0 += pd.Timedelta(1,unit='day')
		tf += pd.Timedelta(1,unit='day')

	axb2.set_ylabel('Daily Runoff\n($mm$ $w.e.$ $d^{-1})$',rotation=270,labelpad=30)
	axb2.set_ylim([28.33,85])
	axb2.set_xlim(Xlims)
	axb2.xaxis.set_visible(False)
	axb2.spines['top'].set_visible(True)
	axb2.spines['bottom'].set_visible(True)
	handlesB = [h3a,h3b]
	axb2.legend(loc='upper right',handles=handlesB,ncol=2)


	# Plot ETI model outputs
	l0,p0 = plot_error_bounds(axs[I_],df_ETI['RdotM(mmWE/hr)'],df_ETI['RdotMvar(mmWE2/hr2)'],\
					  sig_lvl=sig_lvl,color='maroon',alpha=0.33,label='Melting')
	# Plot Rainfall Rates
	h0, = axs[I_].plot(df_ETI['RdotR(mm/hr)'],color='blue',label='Rainfall')
	# Plot Combined Values
	l1,p1 = plot_error_bounds(axs[I_],df_ETI['RdotM(mmWE/hr)']+df_ETI['RdotR(mm/hr)'],df_ETI['RdotMvar(mmWE2/hr2)'],\
					  sig_lvl=sig_lvl,color='black',alpha=0.33,label='Combined')
	axs[I_].set_ylabel('Runoff Rate\n($mm$ $w.e.$ $h^{-1}$)')
	axs[I_].legend(ncol=3,loc='upper left')

	axs[I_].xaxis.set_visible(False)
	axs[I_].spines['top'].set_visible(False)
	axs[I_].spines['bottom'].set_visible(False)
	axs[I_].set_xlim(Xlims)
	axs[I_].set_ylim([-0.1,5*1.05])
	# axs[I_].grid(axis='x',color='gray',linestyle=':')
	y_lims = axs[I_].get_ylim()
	for i_ in range(30):
		dateline = pd.Timestamp('2019-08-01T00:00:00')+i_*pd.Timedelta(1,unit='days')
		axs[I_].plot([dateline,dateline],[y_lims[0]-50,y_lims[1]*1.3],':',color='gray',linewidth=1)

### PLOT SURFACE VELOCITIES & INTERNAL DEFORMATION VELOCITY ###
I_ = 2
if I_ == 2:
	# Get Horizontal Velocity Means and Variances
	S_gap = (df_VS['x_m']/df_VS['x_m'])
	VH_mean = ((df_VS[['x_m','y_m']]**2).sum(axis=1)**0.5)*S_gap
	VH_mean = VH_mean.interpolate(limit_direction='both',limit=30,limit_area='inside')
	VH_var = ((df_VS['x_m']/VH_mean)*df_VS['x_vmm'] + (df_VS['y_m']/VH_mean)*df_VS['y_vmm'])*S_gap
	VH_var = VH_var.interpolate(limit_direction='both',limit=30,limit_area='inside')
	# Get Internal Deformation Velocity Mean and Variance
	S_VI19 = df_VI[df_VI['yr']==2018]['Vi(m a-1)']

	# Plot Results
	# Plot Surface Velocities
	plot_error_bounds(axs[I_],VH_mean*spa,VH_var*spa*spa,\
					  sig_lvl=sig_lvl,color='black',alpha=0.33,label='$V_{surf}$')
	# Plot 24hr rolling average velocity
	VH24 = VH_mean.rolling(pd.Timedelta(24,unit='hr')).mean()
	VH24.index -= pd.Timedelta(12,unit='hr')

	axs[I_].plot(VH24*spa*S_gap,color='dimgrey',label='$\\bar{V}_{surf}$')
	# Plot Internal Deformation Velocity
	axs[I_].plot(Xlims,np.ones(2,)*S_VI19.mean(),color='blue',label='$V_{int}$')
	axs[I_].fill_between(Xlims,np.ones(2,)*(S_VI19.mean() - sig_lvl*S_VI19.std()),\
							   np.ones(2,)*(S_VI19.mean() + sig_lvl*S_VI19.std()),\
							   color='blue',alpha=0.33)
	# Plot Sliding Velocity Fill
	axs[I_].fill_between(VH_mean.index,np.ones(len(VH_mean),)*(S_VI19.mean() + sig_lvl*S_VI19.std()),\
									   VH_mean.values*spa,alpha=0.33,color='maroon',label='$\\hat{V}_{slip}$')
	# Do labels
	axs[I_].set_ylabel('Velocity ($m$ $a^{-1}$)')
	# axs[I_].set_ylim([])
	axs[I_].legend(loc='center left')	
	axs[I_].xaxis.set_visible(False)
	axs[I_].spines['top'].set_visible(False)
	axs[I_].spines['bottom'].set_visible(False)
	axs[I_].set_xlim(Xlims)
	axs[I_].spines['right'].set_visible(False)
	axc2 = axs[I_].twinx()
	# Add cm/d axis
	axc2.set_ylabel('Velocity ($cm$ $d^{-1}$)',rotation=270,labelpad=15)
	axc2.set_ylim(np.array(axs[I_].get_ylim())/365.24e-2)
	# axs[I_].grid(axis='x',color='gray',linestyle=':')
	y_lims = axs[I_].get_ylim()
	axs[I_].set_ylim(y_lims)
	for i_ in range(30):
		dateline = pd.Timestamp('2019-08-01T00:00:00')+i_*pd.Timedelta(1,unit='days')
		axs[I_].plot([dateline,dateline],[y_lims[0]-50,y_lims[1]*1.3],':',color='gray',linewidth=1)

### OVERLAY CONTINUOUS VERTICAL GRID DATELINES ###
Zpad = 0.01
k_ += 1
axB = fig.add_axes([0.1,Y_-(k_)*H_-Zpad,0.8,0.8])

# Work-around for plotting the grid
# for i_ in range(30):
# 	dateline = pd.Timestamp('2019-08-01T00:00:00')+i_*pd.Timedelta(1,unit='days')
# 	axB.plot([dateline,dateline],[Zpad/(0.8+Zpad),1],':',color='gray',linewidth=1)

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
	# if i_ % 2 == 1:
	# axB.text(Xlims[1]+pd.Timedelta(30,unit='hours'),0.99-.335*i_,iLB,fontweight='extra bold')
	# else:
	axB.text(Xlims[0]-pd.Timedelta(40,unit='hours'),0.99-.335*i_,iLB,fontweight='extra bold',fontstyle='italic')



plt.show()

if issave:
	OUT_FILE = 'Fig3_August_Time_Series_%ddpi.png'%(DPI)
	plt.savefig(os.path.join(ODIR,OUT_FILE),dpi=DPI,pad_inches=0,format='png')