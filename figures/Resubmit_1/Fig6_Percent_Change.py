import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
# Figure output directory
ODIR = os.path.join(ROOT,'results','Main','Resubmit_1')
issave = True
DPI = 200

#################################################################################
#### LOAD DATA ##################################################################
#################################################################################

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

plt.rc('font', size=SMALL_SIZE)		  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)	# fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)	# fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)	# legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## Generate Figure
fig = plt.figure(figsize=(11,7))

## Initialize Subplots
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122,sharex=ax0)

flds = ['A(m2)','P(m)','H(m)','Sf','Slp(deg)','Td(Pa)']
c_cycle = {'A(m2)':'black','P(m)':'darkblue','H(m)':'dodgerblue',\
		   'Sf':'maroon','Slp(deg)':'darkorange','Td(Pa)':'red'}
l_dict = {'A(m2)':'A','P(m)':'P','H(m)':'H_{i}','Sf':'S_{f}',\
		  'Slp(deg)':'\\alpha','Td(Pa)':'\\tau_{xz}'}

for f_ in flds:
	idf = df_GUS.T.filter(like=f_).T.sort_values('Dates')
	iDu = idf['mean']
	iDs = idf['stdev']
	iDr = iDu.T.filter(like='1948').values[0]
	# breakpoint()
	ax0.plot(idf['Dates'],100*(iDu - iDr)/iDr,'.-',color=c_cycle[f_],\
				label='$%s$'%(l_dict[f_]))
				# label='$\\Delta %s$ (%s)'%(l_dict[f_],'%'))#,alpha=0.25)

	ax0.plot(idf['Dates'],100*(iDu - iDs - iDr)/iDr,':',color=c_cycle[f_])#,alpha=0.25)
	ax0.plot(idf['Dates'],100*(iDu + iDs - iDr)/iDr,':',color=c_cycle[f_])#,alpha=0.25)

ax0.set_ylabel('Change Relative to 1948 (%)')
ax0.legend(ncol=2,loc='lower left')
ax0.set_xlim([df_GLS['Dates'].min() - pd.Timedelta(300,unit='day'),\
			  df_GLS['Dates'].max() + pd.Timedelta(300,unit='day')])
xlims = ax0.get_xlim()
ylims = ax0.get_ylim()
ax0.text(xlims[0],ylims[0] + (ylims[1] - ylims[0])*0.95,'  a',\
		fontweight='extra bold',fontstyle='italic',fontsize=14)
ax0.grid(axis='y')
ax0.set_xlabel('Year')
ax0.set_title('Upper Sector')

for f_ in flds:
	idf = df_GLS.T.filter(like=f_).T.sort_values('Dates')
	iDu = idf['mean']
	iDs = idf['stdev']
	iDr = iDu.T.filter(like='1948').values[0]
	# breakpoint()
	ax1.plot(idf['Dates'],100*(iDu - iDr)/iDr,'.-',color=c_cycle[f_],\
			label='$%s$'%(l_dict[f_]))#,alpha=0.25)
	ax1.plot(idf['Dates'],100*(iDu - iDs - iDr)/iDr,':',color=c_cycle[f_])#,alpha=0.25)
	ax1.plot(idf['Dates'],100*(iDu + iDs - iDr)/iDr,':',color=c_cycle[f_])#,alpha=0.25)

# ax1.set_ylabel('Change From Average\n$\\Delta$ (%)')
ax1.legend(ncol=2,loc='lower left')
xlims = ax1.get_xlim()
ylims = ax1.get_ylim()
ax1.text(xlims[0],ylims[0] + (ylims[1] - ylims[0])*0.95,'  b',\
		fontweight='extra bold',fontstyle='italic',fontsize=14)
ax1.grid(axis='y')
ax1.set_xlabel('Year')
ax1.set_title('Lower Sector')
plt.show()

if issave:
	OFILE = os.path.join(ODIR,'Fig6_Percent_Change_%ddpi.png'%(DPI))
	plt.savefig(OFILE,dpi=DPI,format='png')   