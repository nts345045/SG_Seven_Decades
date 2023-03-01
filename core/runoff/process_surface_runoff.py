# Import Environmental Packages
import sys
import os
import pandas as pd
import scipy as sp
from scipy.stats import norm
import numpy as np
from glob import glob

# Import Local Supporting Package
import SG_Seven_Decades.util.GPS_utils as gutil
from SG_Seven_Decades.util.DF_utils import rolling_cleanup

import matplotlib.pyplot as plt

### DATA MAPPING SECTION ###
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IWXSG = os.path.join(DROOT,'data','MINDSatUW','MELT','WXSG','Backup-Y.csv')
IABLD = os.path.join(DROOT,'data','MINDSatUW','MELT','LOWER','Ablation_Rate_Logs.csv')
# Output m_f estimate data file (csv format)
OMDATA = os.path.join(DROOT,'processed_data','runoff','melt_factor_estimates.csv')
# Output Summary Data File (csv format)
OSDATA = os.path.join(DROOT,'processed_data','runoff','Surface_Runoff_Rates.csv')

### PROCESSING PARAMETER SECTION ###
# Resampling Parameter
FSW = 60
FSU = 'T'
FSR = str(int(FSW))+FSU

# Degree Day Factor Processing Parameters
T_0 = 1.         # [deg C] Threshold temperature for melt factor calculation
fWE = 0.917      # Water equivalent factor (ablated ice thickness to water column height)



### LOAD DATA ###
# Load Ablation Data
df_ABL = pd.read_csv(IABLD,parse_dates=[8,9])
# Load Weather Station Data
df_WXR = pd.read_csv(IWXSG,parse_dates=True,index_col=['Time'])
print('WX and ablation raw data loaded')

### WEATHER (WX) DATA PREPROCESSING ###
## WX Preprocessing
# Rename columns with shortened names for ease
df_WXR = df_WXR.rename(columns={'Outdoor Temperature(C)':'TEMP(C)','Solar Rad.(w/m2)':'SFLUX(w/m2)','Monthly Rain(mm)':'RAIN(mm)'})
# Evenly sample data to 60-minute intervals
df_WX = df_WXR.copy().resample(FSR).mean()
# Conduct rolling cleanup 
df_WX = rolling_cleanup(df_WX,df_WXR,FSW,FSU,Clip=0.0)

## Hourly Temperature Series
S_temp = df_WX['TEMP(C)']

## Hourly Rainfall Series
# Get hourly incremental rainfall amounts
dRC = df_WX['RAIN(mm)'].values[1:] - df_WX['RAIN(mm)'].values[:-1]
# Divide by sample intervals, just in case the hourly resampling didn't quite take
RdotR = dRC/((df_WX.index[1:] - df_WX.index[:-1]).total_seconds()/3600)
# Add a 0-valued rate at the start of the record (ascribes rates to the end of sampling windows)
RdotR = np.concatenate([np.array([0]),RdotR])
# Create Series to contain Rdot_rain
S_RdotR = pd.Series(RdotR,index=df_WX.index,name='Rdot_rain(mm/hr)')
print('Rainfall rate processing done')

### ABLATION AND MELT FACTOR DATA PROCESSING ###
## Positive Degree Hour Processing
# Create Holder Lists 
TairPDH = []      # [K hr] Holder for Air Temperature Positive Degree Hours 
rTimePDH = []     # [datetime] Holder for ablation window end time
lTimePDH = []     # [datetime] Holder for ablation window end time
dtPDH = []        # [deltatime] Holder for ablation measurement elapsed times
airPDHcum = [0]   # [K hr] Holder for cumulative

# Iterate Across Data
for i_ in np.arange(1,len(S_temp)):
    ts = S_temp.index[i_-1]
    te = S_temp.index[i_] 

    # Get delta time for temperature measurements
    dt = (te - ts).total_seconds()
    # Convert to Hours
    dthr = dt/3600.
    
    # Enforce the threshold temperature (T_0)
    Tair1 = S_temp.values[i_]
    if Tair1 < T_0:
        Tair1 = 0
    
    Tair0 = S_temp.values[i_-1]
    if Tair0 < T_0:
        Tair0 = 0
    
    # Calculate Positive Degree Hours (PDH with trapezoidal approx)
    iPDHair = 0.5*(Tair1 + Tair0)*dthr
    
    # Calculate Cumulative Positive Degree Hours
    TairPDH.append(iPDHair)
    if i_ >1:
        airPDHcum.append(np.nansum([airPDHcum[-1],iPDHair]))
        
    lTimePDH.append(ts)
    rTimePDH.append(te)
    dtPDH.append(dthr)

# Compose the Positive Degree Hour Estimate DataFrame
df_PDH = pd.DataFrame({'lTime':lTimePDH,'rTime':rTimePDH,'dt(hr)':dtPDH,\
                       'PDH':TairPDH,'PDHcum':airPDHcum},index=rTimePDH)

## Ablation Rate and Melt Factor Processing
df_MF = pd.DataFrame()
# Create Melt Factor holder
# Calculate Cumulative Ablation Time-Series for each Station
MF_entries = []
# Iterate Across Station
for s_,sta in enumerate(df_ABL['sta'].unique()):
    idf = df_ABL[(df_ABL['sta']==sta) & (df_ABL['cum_abl']>0)]
    sentries = []
    # Iterate Across Measurement Pairs
    for i_ in range(len(idf)):
        # Get Ablation Measurement Pair Bounding Times
        ts = pd.Timestamp(idf['tstart'].values[i_])
        te = pd.Timestamp(idf['tend'].values[i_])     
        # Fetch the change in surface elevation & convert to mm
        dhice = idf['abl_seg_cm'].values[i_]*10.
        # Convert ablated ice to water column equivalent
        dhh2o = dhice*fWE
        # Fetch Positive Degree Hours
        idf_PDH = df_PDH[(df_PDH.index >= ts) & (df_PDH.index < te)].copy()
        # Calculate cumulative PDH value for incremental ablation value
        iPDH_sums = idf_PDH['PDH'].sum()
        dthrs = (te - ts).total_seconds()/(3600)
        
        ######################################################################
        #### Calculate a series of melt factors for different temperature ####
        S_MF = dhh2o / iPDH_sums #[cm hr**-1 / *K hr]
        ######################################################################
        
        # Create DataFrame Line to append to melt factor model
        ientry = [sta,idf['lat_u'].values[i_],idf['lon_u'].values[i_],idf['elev_u'].values[i_],\
                  idf.index[i_],i_,S_MF,dhice,ts,te]
        # Compile Station Entries
        sentries.append(ientry)
    # Create station-wise intermediate Melt Factor DataFrames (df_MFi)
    df_MFi = pd.DataFrame(sentries,columns=['station','lat','lon','elev',\
                                            'midx','iidx','MF','dhice','ts','te'])
    # Update Index with Observation Index
    df_MFi.index = df_MFi['midx']
    # Concatenate iterative results into report DataFrame
    df_MF = pd.concat([df_MF,df_MFi],axis=0)

print('Melt factor estimates made. Saving to disk')

df_MF.to_csv(OMDATA,header=True,index=True)

### Surface Runoff Supply Rate Processing ###
# Get mean melt-factor value for further analyses
mfbar = df_MF['MF'].mean()
# Calculate melt-produced surface runoff rate
S_Mdot = S_temp*mfbar
S_Mdot.name = 'Mdot(mmWE/hr)'
# Sum melt and rainfall
S_RdotS = S_Mdot + S_RdotR
S_RdotS.name = 'Rdot_surf(mmWE/hr)'

print('Processing complete. Saving summary file to disk')
### Compile and Save Outputs ###
df_summary = pd.concat([S_temp,S_Mdot,S_RdotR,S_RdotS],axis=1,ignore_index=False)

df_summary.to_csv(OSDATA,header=True,index=True)