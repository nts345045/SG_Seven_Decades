"""
STEP3_rotate_ROV1.py - This script rotates GPS displacements and covariance matrices about an origin at u_i = <0,0,0> 

"""

import pandas as pd
import numpy as np
import SG_Seven_Decades.util.GPS_utils as gutil
import os

### Data Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S2C_rov1_despiked.csv')
# Temporary save file location during Covariance Matrix Rotation (takes awhile)
TPATH = os.path.join(DROOT,'processed_data','gps','continuous','tmp','STEP3C_Rotation')
# Output Data File (csv format)
ODATA = os.path.join(DROOT,'processed_data','gps','continuous','S3C_rov1_rotated.csv')


try:
	os.mkdir(TPATH)
except FileExistsError:
	print('tmp/ storage for covariance matrices already exists. Previous data may be overwritten')

### PROCESSING PARAMETERS ###
# Rotation Settings
aa = 7.3*(np.pi/180.)     # alpha - Bed Slope Rotation Correction Factor
bb = 0.					  # beta - Keep things horizontal in the across-flow direction
yy = -34*(np.pi/180.)     # gamma - Flow direction rotation angle

# How often to print out updates on covariance rotation
update = 1e4
# How often to save progressive covariance data from rotation
savepoints = None
rota_kwargs = {'update':update,'savepoints':savepoints,'save_loc':TPATH}


### LOAD DATA ###
df_GPS = pd.read_csv(IDATA,parse_dates=True,index_col='GPST')
print('Data Loaded')


### ROTATE DATA ###
print('Rotating Data')
df_XYZ = gutil.rotate_columns(df_GPS['mE'],df_GPS['mN'],df_GPS['mZ'],aa,bb,yy)
print('Data Rotated!')

breakpoint
### ROTATE COVARIANCE MATRIX ###
print('Rotating Covariance Matrices')
df_cXYZ = gutil.rotate_ENU_cov(df_GPS,aa,bb,yy,**rota_kwargs)
print('Covariance Matrices Rotated!')


print('Concatenating Rotated Data')
df_XYZ = pd.concat([df_XYZ,df_cXYZ],axis=1,ignore_index=False)


### SAVE TO DISK ###
print('Saving rotated data and covariance matrices')
df_XYZ.to_csv(ODATA,header=True,index=True)




