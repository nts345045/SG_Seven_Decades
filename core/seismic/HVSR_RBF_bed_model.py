from scipy.interpolate import Rbf, griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTDIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/INTERPOLATION/Grids'

### GENERATE ICE SURFACE ELEVATION RBF INTERPOLANT
ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VELOCITY'
DG = 1
kwargs={}
# Load Data
df = pd.read_csv(ROOT+'/Campaign_Vels_Q1_data.csv',index_col=0,parse_dates=['t1','t2'])
# Set up data filter
IDX = (df['dt(days)'] >= 4) & (df['c_mbrs'] >= 3)

# Set up grid vectors
xp = df[IDX]['mE'].values
yp = df[IDX]['mN'].values
zp = df[IDX]['mZ'].values
xi = np.arange(df['mE'].min(),df['mE'].max(),DG)
yi = np.arange(df['mN'].min(),df['mN'].max(),DG)
XI,YI = np.meshgrid(xi,yi)


# Create RBF interpolated ice surface elevation interpolate
rbfZs = Rbf(xp,yp,zp,**kwargs)
Zsurf = rbfZs(XI,YI)
# Create surface mask
ZM = griddata((xp,yp),df[IDX]['VmE'].values,(XI,YI))
ZM = ZM/ZM

### GENERATE ICE THICKNESS RBF INTERPOLANT
HV_ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY/OpenHVSR_Project/Outputs'
df = pd.read_csv(HV_ROOT+'/Master_Estimate_1788pm168mps_b1_v3.csv',parse_dates=['start_time','end_time'])
# RBF key-word-arguments holder
kwargs = {}

# Preferred Realizations for Ice-Bed Interface Modeling
exsta_d = {'R107':2,'R112':0,'1115':1,'2116':0,'2130':2,'1117':0,'1118':0,'R108':0,\
		   '1119':0,'1120':3,'1121':0,'1122':1,'1123':3,'1124':1,'1129':0,'1125':0,\
		   '1126':0,'2129':1,'2114':1,'1111':2,'R101':0,'R102':2,'R104':2,'R103':0,\
		   'R105':2,'R106':2,'R108':0,'R110':0,'R112':0,'R109':2,'R111':1,'R107':2,\
		   'R2108':0,'1110':1,'1130':0,'1112':1,'1113':2}

# Generate df for data accumulation
df_HV = pd.DataFrame()

# Pull the preferred curve data from bulk results
for i_,k_ in enumerate(exsta_d.keys()):
	j_ = exsta_d[k_]
	if i_ == 0:
		df_HV = df[df['DAS']==k_].iloc[j_]
	else:
		S_tmp = df[df['DAS']==k_].iloc[j_]
		df_HV = pd.concat([df_HV,S_tmp],axis=1)

# Conduct transpose to right things...
df_HV = df_HV.T

# get coordinates for data
xq = df_HV['mE'].astype(float).values
yq = df_HV['mN'].astype(float).values


# Create Mask
VM = griddata((xq,yq),df_HV['mE'].values,(XI,YI))
VM = VM/VM

## Create Liner Interpolated Surface
VHl = griddata((xq,yq),df_HV['mZ'] - df_HV['H_gau_u'],(XI,YI))

## Create RBF surfaces
# Mean Ice Thickness
rbfHgu = Rbf(xq,yq,df_HV['H_gau_u'].astype(float).values,**kwargs)
VHIu = rbfHgu(XI,YI)
# StDev Ice Thickness
rbfHgo = Rbf(xq,yq,df_HV['H_gau_o'].astype(float).values,**kwargs)
VHIo = rbfHgo(XI,YI)


### CALCULATE ICE-BED INTERFACE ELEVATION FROM RBFs
VZbu = Zsurf - VHIu
VZbo = Zsurf - VHIo



### SAVE SURFACES
# Coordinate Grids
np.save(OUTDIR+'/Easting_Grid.npy',XI)
np.save(OUTDIR+'/Northing_Grid.npy',YI)
# Surface Elevation & Mask
np.save(OUTDIR+'/Surface_Elevation_RBF.npy',Zsurf)
np.save(OUTDIR+'/Surface_Elevation_RBF_MASK.npy',ZM)
# Ice-Bed Interface Elevation & Mask
np.save(OUTDIR+'/HVSR_Bed_Elevation_Mean_RBF.npy',VZbu)
np.save(OUTDIR+'/HVSR_Station_RBF_MASK.npy',VM)
# Ice-Thickness & Uncertainties
np.save(OUTDIR+'/HVSR_Ice_Thickness_Mean_RBF.npy',VHIu)
np.save(OUTDIR+'/HVSR_Ice_Thickness_STD_RBF.npy',VHIo)