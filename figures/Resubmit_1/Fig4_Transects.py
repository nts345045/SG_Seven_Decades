import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# GitHub Repo for the Label Lines package here: https://github.com/cphyc/matplotlib-label-lines
# Pulled 30. August 2021
# from labellines import labelLine, labelLines


"""
TODO
- Get LabelLines onto BigCUDA
- Export gaussian representations of perturbed surfaces
	- Use recursive estimate of mean and variance from MCMC
		e.g., https://math.stackexchange.com/questions/374881/recursive-formula-for-variance

"""
def percent_loc(ax,xpct,ypct):
	xlims = ax.get_xlim()
	ylims = ax.get_ylim()
	xc = xlims[0] + (xlims[1] - xlims[0])*xpct
	yc = ylims[0] + (ylims[1] - ylims[0])*ypct
	return xc,yc


# Root Directory Relative Path
ROOT = os.path.join('..','..','..','..')
# Along Flow Processed Transect Data
AFTD = os.path.join(ROOT,'processed_data','transects','Compiled_AAp_Transect.csv')
# Along Flow WLS estimated Surface Profile Models
AFTM = os.path.join(ROOT,'processed_data','transects','AAp_Transect_Surfaces_WLS_Results.csv')

# Upper Sector Processed Transect Data
USTD = os.path.join(ROOT,'processed_data','transects','Compiled_BBp_Transect.csv')
# Upper Sector polyfit / MCMC Fit Surface Models
USTM = os.path.join(ROOT,'processed_data','transects','BBp_Transect_Surfaces_Poly_MCMC_Results.csv')

# Lower Sector Processed Transect Data
LSTD = os.path.join(ROOT,'processed_data','transects','Compiled_CCp_Transect.csv')
# Lower Sector polyfit / MCMC Fit Surface Models
LSTM = os.path.join(ROOT,'processed_data','transects','CCp_Transect_Surfaces_Poly_MCMC_Results.csv')

# Path To Meier (1957) Extracted Data
M57D = os.path.join(ROOT,'data','MINDSatUW','GIS','Transects','M57')

# Figure output directory
ODIR = os.path.join(ROOT,'results','Main','Resubmit_1')
issave = True
DPI = 200

#### LOAD DATA ####
df_AFD = pd.read_csv(AFTD,index_col=0)
df_AFM = pd.read_csv(AFTM,index_col=0)
df_USD = pd.read_csv(USTD,index_col=0)
df_USM = pd.read_csv(USTM,index_col=0)
df_LSD = pd.read_csv(LSTD,index_col=0)
df_LSM = pd.read_csv(LSTM,index_col=0)


###########################################################################################
##### PLOTTING SECTION ####################################################################
###########################################################################################
mpft = 1/3.2808 	# [m ft-1]
sig_lvl = 1 		# [dimless] Sigma Bound to show for data

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


fig = plt.figure(constrained_layout=True,figsize=(13.,10.))
gs = fig.add_gridspec(2,2)
f_axA = fig.add_subplot(gs[0,:])
f_axB = fig.add_subplot(gs[1,0])
f_axC = fig.add_subplot(gs[1,1])

fmt_cycle = ['-']#,'b--','b-.']
c_cycle=['midnightblue','navy','darkblue','b','royalblue','dodgerblue','cornflowerblue','steelblue','slategray','dimgrey']
s_order=['48','55','66','70','79','86','99','09','18']
endash = u'\u2013'






##################################
#### PLOT ALONG FLOW TRANSECT ####
##################################
handles1 = []; handles = []
for i_,c_ in enumerate(s_order):
	if int(c_) < 40:
		iyr = 2000+int(c_)
		if c_ == '18':
			lyr = '2017/2019'
		else:
			lyr = str(iyr)
	else:
		iyr = 1900+int(c_)
		lyr = str(iyr)

	icolor = c_cycle[i_%len(c_cycle)]

	# Plot Modeled Surface Intercept Points
	f_axA.plot(df_AFM.filter(like=c_).filter(like='b').filter(like='mean'),color=icolor)

	# Plot Surface Models
	iMu = df_AFM.filter(like=c_).filter(like='b').filter(like='mean')
	iMs = df_AFM['b%s_var'%(c_)]**0.5
	h_, = f_axA.plot(iMu,label=lyr,color=icolor)
	handles1.append(h_)
	f_axA.fill_between(iMu.index,iMu.values.reshape(len(iMu),) - sig_lvl*iMs.values.reshape(len(iMu),),\
								 iMu.values.reshape(len(iMu),) + sig_lvl*iMs.values.reshape(len(iMu),),\
								 color=icolor,alpha=0.25)
	# Plot Surface Data
	iDu = df_AFD.filter(like=c_).filter(like='Surf').filter(like='mean')
	iDs = df_AFD.filter(like=c_).filter(like='Surf').filter(like='std')
	f_axA.errorbar(iDu.index,iDu.values.reshape(len(iDu),),\
							 yerr=sig_lvl*iDs.values.reshape(len(iDu),),\
							 color=icolor,fmt='.',capsize=2)

## Plot HV Elevations
iDu = df_AFD['Bed18_mean']
iDs = df_AFD['Bed18_std']
IND = iDu.index < 6500
h_ = f_axA.errorbar(iDu[IND].index,iDu[IND].values.reshape(len(iDu[IND]),),\
						 yerr=sig_lvl*iDs[IND].values.reshape(len(iDu[IND]),),\
						 color='maroon',fmt='o',capsize=5,label='HV Bed Elevations')
handles.append(h_)
h_ = f_axA.errorbar(0,0,yerr=0,color='maroon',fmt='s',capsize=5,label='Exposed Elevations')
handles.append(h_)
h_, = f_axA.plot([0,0],[0,1],color='maroon',label='HV Valley Profiles')
handles.append(h_)


## Plot Meier Bed Profile
# Path to Shotpoint Locations
M57S = os.path.join(M57D,'Meier_1957_Bed_Reflections.csv')

# Manual Shift Factors & Transect Intersections
df_M57BR = pd.read_csv(M57S,index_col=0)

## Bedrock Polyline Extracted using WebPlotDigitizer
df_M57_BP = pd.read_csv(os.path.join(M57D,'Meier_1957_Bed_Polyline.csv'),header=None,index_col=0)*mpft
# Shift from ft --> m & format
df_M57_BP.index = df_M57_BP.index*mpft
df_M57_BP.index.name = 'LinDist(m)'
df_M57_BP = df_M57_BP.rename(columns={1:'BE52(m)'})

## Reflection Point Locations Along Bed
df_M57_SP = pd.read_csv(os.path.join(M57D,'Meier_1957_Bed_Reflections.csv'),header=None,index_col=0)*mpft
# Shift from ft --> m & format
df_M57_SP.index = df_M57_SP.index*mpft
df_M57_SP.index.name = 'LinDist(m)'
df_M57_SP = df_M57_SP.rename(columns={1:'BE52(m)ref'})

### Get Correction Lateral Shift From Reference Points
idx_sp3 = df_M57_SP.index[2]
# Upper HVSR in LS is 360m up-glacier from SP03
idx_c52 =  (5600 + 360) - idx_sp3
df_M57_BP.index = df_M57_BP.index + idx_c52
df_M57_SP.index = df_M57_SP.index + idx_c52
AFT_c52 = 5960 - df_M57BR.index[2]
### Get locations of US & LS
USpX = df_M57_SP.index[1]
LSpX = 6200

h_, = f_axA.plot(df_M57_SP,'k^',label='M57 Bed Elevations',markersize=8)
handles.append(h_)
h_, = f_axA.plot(df_M57_BP,'--',color='dimgrey',label='M57 Valley Profiles')
handles.append(h_)

for i_ in handles1:
	handles.append(i_)

# Plot Transect Intercepts
f_axA.plot(np.ones(2)*USpX,[1700,2400],'r:')
f_axA.text(USpX+20,2300,"B{0}B'".format(endash),fontweight='extra bold',color='r')

f_axA.plot(np.ones(2)*LSpX,[1700,2400],'r:')
f_axA.text(LSpX+20,2300,"C{0}C'".format(endash),fontweight='extra bold',color='r')

f_axA.set_xlim(3000,9400)
f_axA.set_ylim(1700,2350)

f_axA.legend(handles=handles,ncol=3)
endash = u'\u2013'
f_axA.set_xlabel("Distance Along A{0}A' (m)".format(endash))
f_axA.set_ylabel("Elevation (m a.s.l.)")

xc,yc = percent_loc(f_axA,0.01,0.025)
f_axA.text(xc,yc,'a',fontsize=14,fontweight='extra bold',fontstyle='italic')


####################################
#### PLOT UPPER SECTOR TRANSECT ####
####################################
for i_,c_ in enumerate(s_order):
	if int(c_) < 40:
		iyr = 2000+int(c_)
		if c_ == '18':
			iyr = 2017
			lyr = '2017'
		else:
			lyr = str(iyr)
	else:
		iyr = 1900+int(c_)
		lyr = str(iyr)
	icolor = c_cycle[i_%len(c_cycle)]

	# Plot Surface Models
	iMu = df_USM.filter(like=c_).filter(like='Surf').filter(like='rmean')
	iMs = df_USM.filter(like=c_).filter(like='Surf').filter(like='var').abs()**0.5
	f_axB.plot(iMu,label=iyr,color=icolor)
	# f_axB.fill_between(iMu.index,iMu.values.reshape(len(iMu),) - sig_lvl*iMs.values.reshape(len(iMu),),\
	# 							 iMu.values.reshape(len(iMu),) + sig_lvl*iMs.values.reshape(len(iMu),),\
	# 							 color=icolor,alpha=0.25)

	# Plot Surface Data
	iDu = df_USD.filter(like=c_).filter(like='Surf').filter(like='mean')
	iDs = df_USD.filter(like=c_).filter(like='Surf').filter(like='std')
	f_axB.errorbar(iDu.index,iDu.values.reshape(len(iDu),),\
							 yerr=sig_lvl*iDs.values.reshape(len(iDu),),\
							 color=icolor,fmt='.',capsize=2)

### Plot Bed Features ###
## New Bed Data & Model

IND = (df_USD['Bed18_mean'].notna()) | (df_USD['ExposedBed_mean'].notna())
iDu = df_USD.filter(like='Bed').filter(like='mean').sum(axis=1)[IND]
iDs = df_USD.filter(like='Bed').filter(like='std').sum(axis=1)[IND]
IND = (df_LSM.index >= iDu.index.min()) & (df_LSM.index <= iDu.index.max())
f_axB.plot(df_USM['Bed_rmean(m asl)'][IND],label='Bed',color='maroon')
f_axB.errorbar(df_USD['Bed18_mean'].index.values.reshape(len(df_USD),),\
			   df_USD['Bed18_mean'].values.reshape(len(df_USD),),\
			   yerr=sig_lvl*df_USD['Bed18_std'].values.reshape(len(df_USD),),\
			   color='maroon',fmt='o',capsize=5)
f_axB.errorbar(df_USD['ExposedBed_mean'].index.values.reshape(len(df_USD),),\
			   df_USD['ExposedBed_mean'].values.reshape(len(df_USD),),\
			   yerr=sig_lvl*df_USD['ExposedBed_std'].values.reshape(len(df_USD),),\
			   color='maroon',fmt='s',capsize=5)

## Meier (1957) Features
US_offset = 985.
df_M57_US_BP = pd.read_csv(os.path.join(M57D,'Compiled_Transect_IV_Bed_Topography.csv'),index_col='x(m)')

f_axB.plot(df_M57_US_BP.index*-1+80+US_offset,df_M57_US_BP['z(m)'],'--',color='dimgrey')

# Plot Meier (1957) Reflection Points
M57_Ref = np.array([[4573.480176211459, 6413.333333333334],\
					[5860.704845814983, 6113.333333333336],\
					[7262.79001468429, 6386.66666666667]])
M57_Ref /= 3.2808
f_axB.plot(M57_Ref[:,0] - M57_Ref[1,0] +40 + US_offset,M57_Ref[:,1],'k^',markersize=8)


# Plot Transect Intercepts
f_axB.plot(np.zeros(2)+ US_offset,[1700,2400],'r:')
f_axB.text(20 + US_offset,2300,"A{0}A'".format(endash),fontweight='extra bold',color='r')

# Do Axes
f_axB.set_ylabel('Elevation (m a.s.l.)')
f_axB.set_xlabel("Distance Along B{0}B' (m)".format(endash))
f_axB.set_ylim(1700,2350)

xc,yc = percent_loc(f_axB,0.01,0.025)
f_axB.text(xc,yc,'b',fontsize=14,fontweight='extra bold',fontstyle='italic')






####################################
#### PLOT LOWER SECTOR TRANSECT ####
####################################
for i_,c_ in enumerate(s_order):
	if int(c_) < 40:
		iyr = 2000+int(c_)
		if c_ == '18':
			iyr = 2019
			lyr = '2019'
		else:
			lyr = str(iyr)
	else:
		iyr = 1900+int(c_)
		lyr = str(iyr)
	icolor = c_cycle[i_%len(c_cycle)]

	# Plot Surface Models
	iMu = df_LSM.filter(like=c_).filter(like='Surf').filter(like='rmean')
	iMs = df_LSM.filter(like=c_).filter(like='Surf').filter(like='var').abs()**0.5
	f_axC.plot(iMu,label=iyr,color=icolor)
	# f_axC.fill_between(iMu.index,iMu.values.reshape(len(iMu),) - sig_lvl*iMs.values.reshape(len(iMu),),\
	# 							 iMu.values.reshape(len(iMu),) + sig_lvl*iMs.values.reshape(len(iMu),),\
	# 							 color=icolor,alpha=0.25)

	# Plot Surface Data
	iDu = df_LSD.filter(like=c_).filter(like='Surf').filter(like='mean')
	iDs = df_LSD.filter(like=c_).filter(like='Surf').filter(like='std')
	f_axC.errorbar(iDu.index,iDu.values.reshape(len(iDu),),\
							 yerr=sig_lvl*iDs.values.reshape(len(iDu),),\
							 color=icolor,fmt='.',capsize=2)

### Plot Bed Features ###
## New Bed Data & Model

IND = (df_LSD['Bed18_mean'].notna()) | (df_LSD['ExposedBed_mean'].notna())
iDu = df_LSD.filter(like='Bed').filter(like='mean').sum(axis=1)[IND]
iDs = df_LSD.filter(like='Bed').filter(like='std').sum(axis=1)[IND]
IND = (df_LSM.index >= iDu.index.min()) & (df_LSM.index <= iDu.index.max())
f_axC.plot(df_LSM['Bed_rmean(m asl)'][IND],label='Bed',color='maroon')
f_axC.errorbar(df_LSD['Bed18_mean'].index.values.reshape(len(df_LSD),),\
			   df_LSD['Bed18_mean'].values.reshape(len(df_LSD),),\
			   yerr=sig_lvl*df_LSD['Bed18_std'].values.reshape(len(df_LSD),),\
			   color='maroon',fmt='o',capsize=5)
f_axC.errorbar(df_LSD['ExposedBed_mean'].index.values.reshape(len(df_LSD),),\
			   df_LSD['ExposedBed_mean'].values.reshape(len(df_LSD),),\
			   yerr=sig_lvl*df_LSD['ExposedBed_std'].values.reshape(len(df_LSD),),\
			   color='maroon',fmt='s',capsize=5)

## Meier (1957) Features
LS_offset = 1080.
df_M57_LS_BP = pd.read_csv(os.path.join(M57D,'Compiled_Transect_III_Bed_Topography.csv'),index_col='x(m)')

f_axC.plot(df_M57_LS_BP.index*-1+80+LS_offset,df_M57_LS_BP['z(m)'],'--',color='dimgrey')

f_axC.plot(80 + LS_offset,1830,'k^',markersize=8)

# Plot Transect Intercepts
f_axC.plot(np.zeros(2) + LS_offset,[1700,2400],'r:')
f_axC.text(20 + LS_offset,2300,"A{0}A'".format(endash),fontweight='extra bold',color='r')

# Label Axes
f_axC.set_ylabel('Elevation (m a.s.l.)')
f_axC.set_xlabel("Distance Along C{0}C' (m)".format(endash))
f_axC.set_ylim(1700,2350)

xc,yc = percent_loc(f_axC,0.01,0.025)
f_axC.text(xc,yc,'c',fontsize=14,fontweight='extra bold',fontstyle='italic')

plt.show()

if issave:
	OFILE = os.path.join(ODIR,'Fig4_Transects_%ddpi.png'%(DPI))
	plt.savefig(OFILE,dpi=DPI,format='png')   