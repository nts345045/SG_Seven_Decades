import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### HELPER FUNCTIONS ###
def pct_rng(val_list,pct=0.9):
	imax = np.nanmax(val_list)
	imin = np.nanmin(val_list)
	irng = imax - imin
	return imin + irng*pct

def deg2rad(deg):
	return deg*(np.pi/180)


### PROCESSING FUNCTIONS ###
def calculate_Vint(nn,Sf,aa,BB,Hi,pp,gg):
	"""
	Use the Nye (1952) flowlaw equation to estimate the 
	vertically integrated internal deformation rate
	:: INPUTS ::
	:param nn: flow-law exponent [3,4], usually
	:param Sf: valley shape-factor, falls within [0.5,1.0]
	:param aa: ice-surface slope [rad]
	:param BB: effective viscosity [Pa yr**(-1/nn)]
	:param Hi: ice column thickness [m]
	:param pp: ice density [kg m**-3]
	:param gg: 

	"""
	return (2/(1+nn))*Hi**(nn+1)*(((Sf*gg*pp*np.sin(aa))/BB))**nn


issave = False
ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FLOW_MODELS'

# Load Viscosity Inversion Results
df_Bmod = pd.read_csv(ROOT+'/B_Inversion/Inversion_Results_LOOCV.csv')
# Load Geometric Parameters
df_BBp = pd.read_csv(ROOT+'/BBp_Geometric_Parameters_200m_SW_v2.csv',index_col='yr')
df_CCp = pd.read_csv(ROOT+'/CCp_Geometric_Parameters_200m_SW_v2.csv',index_col='yr')
# Load Surface Velocity Observations
df_us = pd.read_csv(ROOT+'/Surface_Velocities.csv',index_col='Year')

# Set Flowlaw Constants
nn = 3
gg = 9.81
pp = 916.7
BB = df_Bmod['B'].mean()

# Calculate the Driving Stresses
Txz_BBp = df_BBp['Sf']*gg*pp*np.sin(df_BBp['alpha']*(np.pi/180))*df_BBp['Hi']
Txz_CCp = df_CCp['Sf']*gg*pp*np.sin(df_CCp['alpha']*(np.pi/180))*df_CCp['Hi']

# Calculate Internal Deformation Velocities at Thalweg
SIA_BBp_t = calculate_Vint(nn,df_BBp['Sf'],df_BBp['alpha']*(np.pi/180),BB,df_BBp['Hi'],pp,gg)
SIA_CCp_t = calculate_Vint(nn,df_CCp['Sf'],df_CCp['alpha']*(np.pi/180),BB,df_CCp['Hi'],pp,gg)

# SIA_BBp_t = (2/(1+nn))*(((df_BBp['Sf']*gg*pp*np.sin(df_BBp['alpha']*(np.pi/180)))/BB))**nn * df_BBp['Hi']**(nn+1)
# SIA_CCp_t = (2/(1+nn))*(((df_CCp['Sf']*gg*pp*np.sin(df_CCp['alpha']*(np.pi/180)))/BB))**nn * df_CCp['Hi']**(nn+1)


# Generate Figure Object
fig = plt.figure(figsize=(8.2,8.8))
# Specify Axes for plotting

mbw = 0.1

ax1 = fig.add_axes([0.1,0.55,0.8,0.35])
ax2 = fig.add_axes([0.1,0.05,0.8,0.35],sharex=ax1)
ax3 = fig.add_axes([0.1,0.40,0.8,0.15],sharex=ax1)

# Create iterable list of handles for convenience
axs = [ax1,ax2,ax3]

# Mute ax1 xlabels
# ax1.xaxis.set_visible(False)

# Make ax3 frame transparent
ax3.patch.set_alpha(0)
for SL_ in ['top','bottom','left','right']:
	ax3.spines[SL_].set_visible(False)
ax3.yaxis.set_visible(False)
ax3.xaxis.set_visible(False)

# Group into axs list
axs = [ax1,ax2,ax3]



# # PLOT
# fig,axs = plt.subplots(nrows=2,sharex=True,figsize=(8.2,8.8))



# Plot Internal Deformation Trends
axs[0].plot(SIA_BBp_t,'o-',label='$V_{INT}$')
axs[1].plot(SIA_CCp_t,'o-',label='$V_{INT}$')

midx = ['r^','bv','ks']

# Bounce between sectors
for i_,c_ in enumerate(['bbp','ccp']):
	# Parse season
	for j_,s_ in enumerate(['Summer','Winter','Annual']):
		ifilt = df_us['season']==s_
		udf = df_us[ifilt][c_+'_mean']
		odf = df_us[ifilt][c_+'_std']
		ldf = df_us[ifilt]['Symbol'].values
		axs[i_].errorbar(udf.index,udf,yerr=odf,fmt=midx[j_],capsize=5,label='$V_{SURF}$ '+s_)
		# if sum(udf.notna()) > 0:
		# 	for k_ in range(len(ldf)):
		# 		if udf.notna().iloc[k_] and pd.notna(ldf[k_]):
		# 			axs[i_].text(udf.index[k_]+2,udf.iloc[k_],\
		# 						 '%s\n%s'%(ldf[k_].split('n')[0],ldf[k_].split('n')[1]),\
								 # va='center')
		if sum(udf.notna()) > 0:
			for k_ in range(len(ldf)):
				if udf.notna().iloc[k_] and pd.notna(ldf[k_]):
					if ldf[k_].split('n')[0] == 'S21':
						axs[i_].text(udf.index[k_]+2,udf.iloc[k_],\
						 			'%s\n%s'%(ldf[k_].split('n')[0],ldf[k_].split('n')[1]),\
						 			va='center')
					else:
						axs[i_].text(udf.index[k_]+2,udf.iloc[k_],\
								 '%s'%(ldf[k_].split('n')[0]),\
								 va='center')

# Insert Warming & Cooling Labels
dts = [1945,1970,1980,2020]
axs[2].fill_between([dts[0],dts[1]],[0.1,0.1],[0.5,0.5],color='gray',alpha=0.5)
axs[2].fill_between([dts[1],dts[2]],[0.1,0.1],[0.5,0.525],color='firebrick',alpha=0.25)
axs[2].fill_between([dts[2],dts[3]],[0.1,0.1],[0.525,0.8],color='firebrick',alpha=0.5)
axs[2].text(1948,0.35,'Stable Temperatures',fontweight='extra bold')
axs[2].text(1990,0.35,'Warming Temperatures',fontweight='extra bold')

# Window Dressing


axs[1].legend(loc='lower left')
axs[0].set_ylabel('Velocity (m/yr)')
axs[1].set_ylabel('Velocity (m/yr)')
axs[1].set_xlabel('Year')
axs[0].set_ylim([30,90])
axs[1].set_ylim([0,70])
axs[2].set_ylim([0,1])
axs[0].set_xlim([1945,2029])

for i_,l_ in enumerate(['A','B']):
	iX = pct_rng(axs[i_].get_xlim(),pct=0.015)
	iY = pct_rng(axs[i_].get_ylim(),pct=0.90)
	axs[i_].text(iX,iY,l_,fontweight='extra bold')

plt.show()

if issave:
	SAVE_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/v12_render'
	DPI = 200
	plt.savefig('%s/Figure_7_MultiDecadal_Glacier_Dynamics_and_Climate_%ddpi.png'%(SAVE_DIR,DPI),\
				dpi=DPI,pad_inches=0,format='png')