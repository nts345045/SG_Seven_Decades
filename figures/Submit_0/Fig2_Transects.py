import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# GitHub Repo for the Label Lines package here: https://github.com/cphyc/matplotlib-label-lines
# Pulled 30. August 2021
from labellines import labelLine, labelLines

"""
Create plots for the 

"""
#### RESAMPLING & SMOOTHING OPERATORS ####
def sdfr(df,di,**kwargs):
	"""
	sPATIAL dATAfRAME rESAMPLING - Resample data in a dataframe to a regularly spaced index
	This acts as a wrapper for numpy.interp() invoked for each column in the input DataFrame.
	It is assumed that the index of the input DataFrame is some spatial or similar, monotonically increasing)
	sequence of float values 

	This stands in for the df.resample() method for Timeindex'd DataFrames (df) in the general distribution of Pandas

	:: INPUTS :::
	:type df: pandas.DataFrame
	:param df: Target dataframe to resample
	:type di: float
	:param di: index sampling distance
	:type **kwargs: dictionary
	:param **kwargs: Key Word Arguments for numpy.interp()
	
	:: OUTPUTS ::
	:rtype dfo: pandas.DataFrame
	:return dfo: output DataFrame
	"""
	# Create copy of DataFrame to preserve input
	idf = df.copy()
	# Create new index
	IDX = np.arange(df.index.min(),df.index.max(),di)
	# Ensure that the last sample is bracketed by an interpolation point
	if IDX[-1] != df.index.max():
		IDX = np.append(IDX,[IDX[-1]+di],axis=0)
	# Create OutputDict
	OD = {}
	for c_ in idf.columns:
		# Do interpolation
		ix = np.interp(IDX,idf.index.values,idf[c_].values,**kwargs)
		# Clip ends
		ix[(IDX >= idf[idf[c_].notna()].index.min()) & (IDX <= idf[idf[c_].notna()].index.min())]
		OD.update({c_:ix})

	dfo = pd.DataFrame(OD,index=IDX)

	return dfo

def crm(df,wl=3,n=1,method=np.nanmean):
	idf = df.copy()
	dx = idf.index.values[1] - idf.index.values[0]
	for i_ in range(n):
		# Do Rolling Application of Method
		jdf = idf.rolling(wl).apply(method)
		# Center Window Output
		jdf.index = idf.index.values - (wl*dx)/2
		# Bring Data Back In That Was Clipped
		jdf = jdf.fillna(idf)
	return jdf

#### PHYSICS & GEOMETRIC PROCESSING FUNCTIONS ####
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

def get_Sf_rs(bx,by,sx,sy,bpo,spo,di=10,bywt=None,cov=False):
	"""
	Calculate the valley shape-factor given data for surface and filled-contact
	reference points

	:: INPUTS ::
	:type bx: list-like
	:param bx: X-coordinates of filled-contact data
	:type by: list-like
	:param by: Y-coordinates of filled-contact data
	:type sx: list-like
	:param sx: X-coordinates of fill surface data
	:type sy: list-like
	:param sy: Y-coordinates of fill surface data
	:type bpmo: int
	:param bpmo: filled-contact polynomial model fit order
	:type spmo: int
	:param spmo: fill surface polynomial model fit order
	:type dx: float
	:param dx: Discretization for filled-contact arclength estimation
	:type bywt: None or array-like
	:param bywt: weights for numpy.polyfit(), should be 1/sigma

	:: OUTPUTS ::
	:rtype out: dict
	:return out: dictionary containing the parameters to calculate Sf = A/(P*H)
				 the polynomial coefficients for the surface and bed model
				 and the x-axis locations for model margins and the location of 
				 the thickest modeled ice.
	"""

	# Polyfit for bed data
	pbc = np.polyfit(bx,by,bpo,w=bywt)
	pb = np.polynomial.Polynomial(pbc[::-1])
	polyB = np.poly1d(pbc)
	# Polyfit for surface data
	psc = np.polyfit(sx,sy,spo)
	ps = np.polynomial.Polynomial(psc[::-1])
	polyS = np.poly1d(psc)
	# Difference Polynomials
	pA = (ps - pb)
	roots = pA.roots()
	# Get real-valued, positive roots
	roots = np.abs(roots[(np.isreal(roots)) & (roots > 0)])
	# Intercepts of the surface curve and the bed curve
	x1 = roots[0]
	x2 = roots[1]
	# Estimate Cross-Sectional Area by integrating differenced polynomials at intercepts
	AA = np.diff(np.polynomial.polynomial.polyval(roots,pA.integ(lbnd=x1).coef))[0]
	# Get maximum thickness using derivative of differenced polynomials between roots
	Hroots = pA.deriv().roots()
	Hroot = np.abs(Hroots[(Hroots > x1) & (Hroots < x2)])[0]
	HH = np.polynomial.polynomial.polyval(Hroot,pA.coef)
	# Do Riemann Approximation for Bed Arclength
	xv = np.arange(x1,x2,di)
	yv = polyB(xv)
	PP = np.trapz(np.sqrt(1. + np.gradient(yv,xv)**2),xv)
	out = {'Sf':AA/(HH*PP),'A':AA,'P':PP,'H':HH,'bed_poly':pbc,'surf_poly':psc,'x_edges':roots,'x_thick':Hroot}
	return out

def percent_locs(ax,xpct=0.05,ypct=0.05):
	XL = ax.get_xlim()
	YL = ax.get_ylim()
	X = np.sum(np.array(XL)*np.array([1-xpct,xpct]))
	Y = np.sum(np.array(YL)*np.array([1-ypct,ypct]))
	return X,Y



ft2m = 1/3.2808

# SCL = {'CCp':}
ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY'
# Get HVSR Results
HVSR_file = ROOT+'/OpenHVSR_Project/Outputs/Master_Estimate_Vs1769pm168_b1_v2.csv'

PBO_B = 3
PSO_B = 1
PBO_C = 3
PSO_C = 1


#### GET BEDROCK MODEL AND DATA FROM Meier (1957)
### Get A-A' Data / Along-Flow Elevation Profile

## Bedrock Polyline Extracted using WebPlotDigitizer
df_AAp_bed52 = pd.read_csv(ROOT+'/Transects/M57/Meier_1957_Bed_Polyline.csv',header=None,index_col=0)*ft2m
# Shift from ft --> m & format
df_AAp_bed52.index = df_AAp_bed52.index*ft2m
df_AAp_bed52.index.name = 'LinDist(m)'
df_AAp_bed52 = df_AAp_bed52.rename(columns={1:'BE52(m)'})

## Reflection Point Locations Along Bed
df_AAp_br52 = pd.read_csv(ROOT+'/Transects/M57/Meier_1957_Bed_Reflections.csv',header=None,index_col=0)*ft2m
# Shift from ft --> m & format
df_AAp_br52.index = df_AAp_br52.index*ft2m
df_AAp_br52.index.name = 'LinDist(m)'
df_AAp_br52 = df_AAp_br52.rename(columns={1:'BE52(m)ref'})

### Get Correction Lateral Shift From Reference Points
idx_sp3 = df_AAp_br52.index[2]
# Upper HVSR in LS is 360m up-glacier from SP03
idx_c52 =  (5600 + 360) - idx_sp3
df_AAp_bed52.index = df_AAp_bed52.index + idx_c52
df_AAp_br52.index = df_AAp_br52.index + idx_c52



### Get locations of B-B' & C-C'
BBpX = df_AAp_br52.index[1]
CCpX = 6200


#### GET SURFACE ELEVATION PROFILES 
df_AAp_surfs = pd.read_csv(ROOT+'/Transects/Transect_AAp_Smoothed_TM13_dx10_wl60_n2.csv',index_col=0)


#### GET BED ELEVATION PROFILES FROM MODERN HVSR
df_AAp_bed19 = pd.read_csv(ROOT+'/Transects/SGGS/HVSR_AAp_Transect_Data_v2.csv',index_col='LinDist(m)')
df_AAp_z_bed19 = df_AAp_bed19['Elev19'] - df_AAp_bed19['Hi_mean']
HV_mask = df_AAp_bed19['HVSR_Mask'] == 1



# Create column index
sd_map = {}
for c_ in df_AAp_surfs.columns:
	if '10' not in c_:
		if int(c_[4:6]) < 40:
			iyr = 2000+int(c_[4:6])
			if iyr in [2017,2019]:
				iyr = 2019
		else:
			iyr = 1900+int(c_[4:6])
		sd_map.update({c_:iyr})

df_sd_map = pd.DataFrame(sd_map,index=['year']).transpose().sort_values('year',ascending=True)


#### SURFACE PROCESSING PARAMETERS ####
srdx = 10.  # [m] Surface resampling rate
sswl = 600.	# [m] Surface smoothing window length
aewl = 200.  # [m] Slope extraction window length


# Conduct Slope Calculation
df_alpha = pd.DataFrame()
xx = df_AAp_surfs.index.values
for f_ in df_AAp_surfs.columns:
	if '_' not in f_:
		yy = df_AAp_surfs[f_].values
		aa = (-180./np.pi)*np.arctan(np.gradient(yy,xx))
		yr_ = int(f_[4:6])+1900
		if yr_ < 1940:
			yr_ += 100
		adf = pd.DataFrame({yr_:aa})
		adf.index = xx
		df_alpha = pd.concat([df_alpha,adf],axis=1,ignore_index=False)

# Extract Estimates for B-B'
S_alphas_B = df_alpha[np.abs(df_alpha.index.values - 4122) < aewl/2].mean()
S_alphas_B.name = 'alpha'
# Fix Index
as_list = S_alphas_B.index.tolist()
idx = as_list.index(2019)
as_list[idx] = 2017
S_alphas_B.index = as_list

# Hard-wire assessment of elevation data based slope from SaskSeis1
S_xy17 = df_AAp_bed19[df_AAp_bed19['Type']=='SS17']['Elev19']
x17 = S_xy17.index.values
y17 = S_xy17.values
S_alphas_B[2017] = (-180/np.pi)*np.arctan(np.polyfit(x17,y17,1)[0])

# Extract Estimates for C-C'
S_alphas_C = df_alpha[np.abs(df_alpha.index.values - 6200) < aewl/2].mean()
S_alphas_C.name = 'alpha'




## Transect B-B' (Castleguard Sector)
df_B = pd.read_csv(ROOT+'/Transects/Transect_BBp_Compiled.csv',index_col='LinDist(m)')
df_B_bed = pd.read_csv(ROOT+'/Transects/Transect_IV_Exposed_Bedrock_Cleaned.csv',index_col='LinDist(m)')
df_BBp_br52 = pd.read_csv(ROOT+'/Transects/M57/Compiled_Transect_IV_Bed_Topography.csv',index_col='x(m)')
BBp_offset = 985.


HVIDX = df_B_bed['Estd19'].notna()
# Get HVSR Bed Elevations and STD
df_HV_BBp_bed = df_B_bed[HVIDX][['Mask','Elev19','Estd19']]
# Get Exposed Bedrock Elevations and STD
df_GX_BBp_bed = df_B_bed[~HVIDX]
df_BBp_bed17 = df_B['Elev19'] - df_B['Hi_mean']

##########################
###### Process A-A' ######
##########################





##########################
###### Process B-B' ######
##########################
# idf_BB = df_B_bed[df_B_bed.mean(axis=1).notna()]
# bx = idf_BB[(idf_BB['Mask']==1) & (idf_BB.filter(like='Elev').mean(axis=1).notna())].\
# 			filter(like='Elev').mean(axis=1).index.values
# by = idf_BB[(idf_BB['Mask']==1) & (idf_BB.filter(like='Elev').mean(axis=1).notna())].\
# 			filter(like='Elev').mean(axis=1).values
# byo = idf_BB[(idf_BB['Mask']==1) & (idf_BB.filter(like='Elev').mean(axis=1).notna())].\
# 			filter(like='Elev').std(axis=1).values
S_BBp_HV_bu = df_B['Elev19'] - df_B['Hi_mean']
S_BBp_HV_bu.name = 'Elev17'
S_BBp_HV_bo = df_B['Hi_std']
S_BBp_HV_bo.name = 'std17'
df_BBp_HV = pd.concat([S_BBp_HV_bu,S_BBp_HV_bo],axis=1,ignore_index=False)
S_BBp_BR_bu = df_GX_BBp_bed[df_GX_BBp_bed['Mask']==1].filter(like='Elev').mean(axis=1)
S_BBp_BR_bu.name = 'Elev17'
S_BBp_BR_bo = df_GX_BBp_bed[df_GX_BBp_bed['Mask']==1].filter(like='Elev').std(axis=1)
S_BBp_BR_bo.name = 'std17'
df_BBp_BR = pd.concat([S_BBp_BR_bu,S_BBp_BR_bo],axis=1,ignore_index=False)

df_BBp_BED = pd.concat([df_BBp_HV,df_BBp_BR],axis=0,ignore_index=False)
df_BBp_BED = df_BBp_BED[df_BBp_BED['Elev17'].notna()].sort_index()

bx = df_BBp_BED.index
by = df_BBp_BED['Elev17'].values
byo = df_BBp_BED['std17'].values

# sx = df_C[(df_C.index >530) & (df_C.index < 1500)].index.values
ODB = {}
yearsB = {}

I_ = 0
for c_ in df_B.filter(like='Elev').columns:
	# print(c_)
	if '_' not in c_ and '17' not in c_:
		sidx = (df_B[c_].notna()) & (df_B.index >290) & (df_B.index < 1500)
		sx = df_B[sidx].index.values
		sy = df_B[sidx][c_].values
		if '19' in c_[4:6]:
			# breakpoint()
			iout = get_Sf_rs(bx,by,sx,sy,PBO_B,0,bywt=byo)
		else:
			iout = get_Sf_rs(bx,by,sx,sy,PBO_B,PSO_B,bywt=byo)
		yr = int(c_[4:6])
		if yr < 40:
			yr += 2000
			if yr == 2019:
				yr = 2017
		else:
			yr += 1900
		iout.update({'alpha':S_alphas_B[yr]})
		iout.update({'flag':c_})
		ODB.update({yr:iout})


## Transect C-C' (Lower Sector) ##
df_C = pd.read_csv(ROOT+'/Transects/Transect_CCp_Compiled.csv',index_col='LinDist(m)')
df_C_bed = pd.read_csv(ROOT+'/Transects/Transect_III_Exposed_Bedrock_Cleaned.csv',index_col='LinDist(m)')
CCp_offset = 1080.
df_CCp_br52 = pd.read_csv(ROOT+'/Transects/M57/Compiled_Transect_III_Bed_Topography.csv',index_col='x(m)')


##########################
###### Process C-C' ######
##########################
### DO SOME PROCESSING
HVIDX = df_C_bed['std19'].notna()
# Get HVSR Bed Elevations and STD
df_HV_CCp_bed = df_C_bed[HVIDX][['Mask','Elev19','std19']]
# Get Exposed Bedrock Elevations and STD
df_GX_CCp_bed = df_C_bed[~HVIDX]
df_CCp_bed19 = df_C['Elev19'] - df_C['Hi_mean']

S_CCp_HV_bu = df_C['Elev19'] - df_C['Hi_mean']
S_CCp_HV_bu.name = 'Elev19'
S_CCp_HV_bo = df_C['Hi_std']
S_CCp_HV_bo.name = 'std19'
df_CCp_HV = pd.concat([S_CCp_HV_bu,S_CCp_HV_bo],axis=1,ignore_index=False)
S_CCp_BR_bu = df_GX_CCp_bed[df_GX_CCp_bed['Mask']==1].filter(like='Elev').mean(axis=1)
S_CCp_BR_bu.name = 'Elev19'
S_CCp_BR_bo = df_GX_CCp_bed[df_GX_CCp_bed['Mask']==1].filter(like='Elev').std(axis=1)
S_CCp_BR_bo.name = 'std19'
df_CCp_BR = pd.concat([S_CCp_BR_bu,S_CCp_BR_bo],axis=1,ignore_index=False)

df_CCp_BED = pd.concat([df_CCp_HV,df_CCp_BR],axis=0,ignore_index=False)
df_CCp_BED = df_CCp_BED[df_CCp_BED['Elev19'].notna()].sort_index()

cx = df_CCp_BED.index
cy = df_CCp_BED['Elev19'].values
cyo = df_CCp_BED['std19'].values

# sx = df_C[(df_C.index >530) & (df_C.index < 1500)].index.values
ODC = {}
yearsC = {}

I_ = 0
for c_ in df_C.filter(like='Elev').columns:
	if '_' not in c_:
		sidx = (df_C[c_].notna()) & (df_C.index >530) & (df_C.index < 1500)
		sx = df_C[sidx].index.values
		sy = df_C[sidx][c_].values
		if '19' in c_[4:6]:
			iout = get_Sf_rs(cx,cy,sx,sy,PBO_C,0,bywt=cyo)	
		else:
			iout = get_Sf_rs(cx,cy,sx,sy,PBO_C,PSO_C,bywt=cyo)
		yr = int(c_[4:6])
		if yr < 40:
			yr += 2000
			# if yr == 2019:
			# 	yr = 2017
		else:
			yr += 1900
		iout.update({'alpha':S_alphas_C[yr]})
		iout.update({'flag':c_})
		ODC.update({yr:iout})



###########################################################################################
##### PLOTTING SECTION ####################################################################
###########################################################################################
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


for i_,c_ in enumerate(df_sd_map.index):
	if '10' not in c_:
		
		if int(c_[4:6]) < 40:
			iyr = 2000+int(c_[4:6])
			if iyr in [2017,2019]:
				iyr = '2017/2019'
		else:
			iyr = 1900+int(c_[4:6])
		if iyr == '2017/2019':
			X1 = df_AAp_bed19[df_AAp_bed19['Type']=='SS17'].index[-1]
			X2 = df_AAp_bed19[df_AAp_bed19['Type']=='SS19'].index[0]
			IND = (df_AAp_surfs.index >= X1) & (df_AAp_surfs.index <= X2)
			f_axA.plot(df_AAp_surfs[IND][c_],'--',label=iyr,color=c_cycle[i_%len(c_cycle)])
			cc20172019 = c_cycle[i_%len(c_cycle)]
		else:
			f_axA.plot(df_AAp_surfs[c_],fmt_cycle[i_%len(fmt_cycle)],label=iyr,color=c_cycle[i_%len(c_cycle)])
lines = f_axA.get_lines()
lvect = np.linspace(5000,9000,len(lines))
labelLines(lines,xvals=lvect[::-1],drop_label=True,align=False)

f_axA.plot(df_AAp_surfs[df_AAp_surfs.index <= X1]['Elev19'],color=cc20172019)
f_axA.plot(df_AAp_surfs[df_AAp_surfs.index >= X2]['Elev19'],color=cc20172019)
# for i_,l_ in enumerate(lines):
# 	labelLine(l_,lvect[i_],drop_label=True)
	# labelLines(plt.gca().get_lines(),drop_label=True,xvals=np.linspace(5000,9000,10))
# Plot Meier (1957)
f_axA.plot(df_AAp_bed52,'k:',label='Meier (1957) Bedrock Model')
f_axA.plot(df_AAp_br52,'k^',label='Meier (1957) Reflection Points',markersize=8)


# Plot HVSR Results
f_axA.errorbar(df_AAp_z_bed19[HV_mask].index,df_AAp_z_bed19[HV_mask].values,yerr=2*(df_AAp_bed19[HV_mask]['Hi_std']+0.7),\
			fmt='.',color='firebrick',capsize=5,label='HVSR Bed Elevations $\pm 2\sigma$')
# for S_ in df_AAp_bed19[HV_mask]:
# 	plt.text(S_.index,S_.values,S_)

# Plot Surface Features
# f_axA.plot(df_AAp_bed19['Elev19'][df_AAp_bed19['Type'].isin(['SS17','SS19'])],'kv',label='Geophone')
# f_axA.plot(df_AAp_bed19['Elev19'][df_AAp_bed19['Type']=='SG19'],'ko',label='Ablation Sites')
f_axA.plot(df_AAp_bed19['Elev19'],'ko',label='2017/2019 Sites')

# Plot Transect Intercepts
f_axA.plot(np.ones(2)*BBpX,[1700,2400],'r:')
f_axA.text(BBpX+20,2300,"B-B'",fontweight='extra bold',color='r')

f_axA.plot(np.ones(2)*CCpX,[1700,2400],'r:')
f_axA.text(CCpX+20,2300,"C-C'",fontweight='extra bold',color='r')

f_axA.set_xlim(3000,9400)
f_axA.set_ylim(1700,2350)

# Label Axes
f_axA.set_xlabel("Distance Along A-A' (m)")
f_axA.set_ylabel("Elevation (m ASL)")
f_axA.legend()


#### DO PLOT B for B-B' TRANSECT ####
# Plot Ice Surfaces
sblist = np.sort(list(ODB.keys()))
for i_,yr_ in enumerate(sblist):
	ix = ODB[yr_]['x_edges']
	ixv = np.arange(ix[0],ix[1],5)
	f_axB.plot(ixv - BBp_offset,np.poly1d(ODB[yr_]['surf_poly'])(ixv),color=c_cycle[i_%len(c_cycle)])
lines = f_axB.get_lines()
lvect = np.linspace(-500,500,len(lines))
# labelLines(lines,drop_label=True,align=False,fontsize=8,bbox={'alpha':0.5,'facecolor':'w','linestyle':None})

# Plot Bedrock Model
ixv = np.arange(-1000,750,5)
f_axB.plot(ixv,np.poly1d(ODB[yr_]['bed_poly'])(ixv + BBp_offset),'--',color='dimgrey',label='Polynomial Fit Model')

# Plot HVSR with Uncertainty
f_axB.errorbar(df_B.index-BBp_offset,df_B['Elev19'] - df_B['Hi_mean'],\
				yerr=df_B['Hi_std']*2,fmt='.',capsize=5,color='firebrick')
# Plot Exposed Bedrock With Uncertainty
GXB_used = df_GX_BBp_bed[df_GX_BBp_bed['Mask']==1]
# GXB_unused = df_GX_BBp_bed[df_GX_BBp_bed['Mask']==0]
f_axB.errorbar(GXB_used.index - BBp_offset,GXB_used.filter(like='Elev').mean(axis=1),\
				yerr=GXB_used.filter(like='Elev').std(axis=1)*2,fmt='s',capsize=5,color='firebrick',\
				label='Exposed Bedrock Elevations')
# f_axB.errorbar(GXB_unused.index - BBp_offset,GXB_unused.filter(like='Elev').mean(axis=1),\
# 				yerr=GXB_unused.filter(like='Elev').std(axis=1)*2,fmt='s',capsize=5,color='k',\
# 				label='Unused Bedrock Data')
f_axB.legend()#loc='lower right')



# Plot Meier (1957) Bedrock Elevation Profile
f_axB.plot(df_BBp_br52.index*-1 + 80,df_BBp_br52['z(m)'],'k:')
# Plot Meier (1957) Reflection Points
M57_Ref = np.array([[4573.480176211459, 6413.333333333334],\
					[5860.704845814983, 6113.333333333336],\
					[7262.79001468429, 6386.66666666667]])
M57_Ref /= 3.2808
f_axB.plot(M57_Ref[:,0] - M57_Ref[1,0] +40,M57_Ref[:,1],'k^',markersize=8)




# Plot Transect Intercepts
f_axB.plot(np.zeros(2),[1700,2400],'r:')
f_axB.text(20,2300,"A-A'",fontweight='extra bold',color='r')

# Do Axes
f_axB.set_ylabel('Elevation (m ASL)')
f_axB.set_xlabel("Distance Along B-B' (m)")
f_axB.set_ylim(1700,2350)




#### DO PLOT C FOR C-C' TRANSECT ####
# Plot Ice Surfaces
sclist= np.sort(list(ODC.keys()))
for i_,yr_ in enumerate(sclist):
	ix = ODC[yr_]['x_edges']
	ixv = np.arange(ix[0],ix[1],5)
	f_axC.plot(ixv - CCp_offset,np.poly1d(ODC[yr_]['surf_poly'])(ixv),label=yr_,color=c_cycle[i_%len(c_cycle)])
lines = f_axC.get_lines()
lvect = np.linspace(-800,500,len(lines))
# labelLines(lines,drop_label=True,align=False,fontsize=8,bbox={'alpha':0.5,'facecolor':'w','linestyle':None})

# Plot Bedrock Model
ixv = np.arange(-1000,750,5)
f_axC.plot(ixv,np.poly1d(ODC[yr_]['bed_poly'])(ixv + CCp_offset),'--',color='dimgrey',label='Bed Model')


# Plot HVSR with Uncertainty
f_axC.errorbar(df_C.index-CCp_offset,df_C['Elev19'] - df_C['Hi_mean'],\
				yerr=df_C['Hi_std']*2,fmt='.',capsize=5,color='firebrick')
# Plot Exposed Bedrock With Uncertainty
# f_axC.errorbar(df_GX_CCp_bed.index - CCp_offset,df_GX_CCp_bed.filter(like='Elev').mean(axis=1),\
# 				yerr=df_GX_CCp_bed.filter(like='Elev').std(axis=1)*2,fmt='s',capsize=5,color='k')
# Plot Exposed Bedrock With Uncertainty
GXC_used = df_GX_CCp_bed[df_GX_CCp_bed['Mask']==1]
GXC_unused = df_GX_CCp_bed[df_GX_CCp_bed['Mask']==0]
f_axC.errorbar(GXC_used.index - CCp_offset,GXC_used.filter(like='Elev').mean(axis=1),\
				yerr=GXC_used.filter(like='Elev').std(axis=1)*2,fmt='s',capsize=5,color='firebrick')
# f_axC.errorbar(GXC_unused.index - CCp_offset,GXC_unused.filter(like='Elev').mean(axis=1),\
# 				yerr=GXC_unused.filter(like='Elev').std(axis=1)*2,fmt='s',capsize=5,color='k')

# Plot Meier (1957) Bedrock Elevation Profile
f_axC.plot(df_CCp_br52.index*-1+80,df_CCp_br52['z(m)'],'k:')
f_axC.plot(80,1830,'k^',markersize=8)

# Plot Transect Intercepts
f_axC.plot(np.zeros(2),[1700,2400],'r:')
f_axC.text(20,2300,"A-A'",fontweight='extra bold',color='r')


f_axC.set_ylabel('Elevation (m ASL)')
f_axC.set_xlabel("Distance Along C-C' (m)")
f_axC.set_ylim(1700,2350)





f_axA_X = f_axA.get_xlim()
f_axA_Y = f_axA.get_ylim()
X,Y = percent_locs(f_axA,xpct=0.025,ypct=0.025)
f_axA.text(X,Y,'A',fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')
X,Y = percent_locs(f_axA,xpct=0.975,ypct=0.025)
f_axA.text(X,Y,"A'",fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')


# Put In Annotations
X,Y = percent_locs(f_axB,xpct=0.025,ypct=0.95)
f_axB.text(X,Y,'B',fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')
X,Y = percent_locs(f_axB,xpct=0.975,ypct=0.95)
f_axB.text(X,Y,"B'",fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')
# Label Figure
X,Y = percent_locs(f_axB,xpct=0.95,ypct=0.025)
f_axB.text(X,Y,'Castleguard Sector\nMeier (1957) Transect IV',horizontalalignment='right')



X,Y = percent_locs(f_axC,xpct=0.025,ypct=0.95)
f_axC.text(X,Y,'C',fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')
X,Y = percent_locs(f_axC,xpct=0.975,ypct=0.95)
f_axC.text(X,Y,"C'",fontweight='extra bold',color='k',fontsize=18,horizontalalignment='center')
# Label Figure
X,Y = percent_locs(f_axC,xpct=0.025,ypct=0.025)
f_axC.text(X,Y,'Lower Sector\nMeier (1957) Transect III offset %dm up-glacier'%\
		   (CCpX - df_AAp_br52.index[2]))


# f_axC.legend()
plt.show()

plt.savefig('/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/Figure_2_Valley_Transect_200dpi.png',\
			dpi=200,format='png',pad_inches=0.05)



#### OUTPUT VALLEY GEOMETRIES ####
df_OUT_B = pd.DataFrame()
for yr_ in np.sort(list(ODB.keys())):
	S_tmp = pd.Series(ODB[yr_])
	S_tmp.name = yr_
	df_OUT_B = pd.concat([df_OUT_B,S_tmp[['Sf','alpha','A','P','H']]],axis=1)

df_OUT_B = df_OUT_B.transpose()
df_OUT_B.index.name = 'year'

df_OUT_B.to_csv('/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/Table1_BBp_data.csv',index=True,header=True)

df_OUT_C = pd.DataFrame()
for yr_ in np.sort(list(ODC.keys())):
	S_tmp = pd.Series(ODC[yr_])
	S_tmp.name = yr_
	df_OUT_C = pd.concat([df_OUT_C,S_tmp[['Sf','alpha','A','P','H']]],axis=1)

df_OUT_C = df_OUT_C.transpose()
df_OUT_C.index.name = 'year'

df_OUT_C.to_csv('/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/Table1_CCp_data.csv',index=True,header=True)
