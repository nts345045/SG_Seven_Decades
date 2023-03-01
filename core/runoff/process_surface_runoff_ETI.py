# Import Environmental Packages
import sys
import os
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from copy import deepcopy




### PROCESSING PARAMETER SECTION ###
# Degree Day Factor Processing Parameters
T0 = 0.                             # [deg C] Threshold temperature for melt factor calculation
fWE = 0.910                         # Water equivalent factor 
                                    # (ablated ice thickness to water column height)
SDT = pd.Timedelta(4,unit='hour')   # Smoothing window length for temperature & shortwave

## INVERSION PARAMETER CONTROL BLOCK
# Input Data Filtering Melt Rate Estimate Data (require minimum data coverage & sampling periods)
min_fract = 0.95    # [fract] Minimum data coverage from WX data for M estimate 
min_DT = 24*4       # [hrs] Minimum time between ablation measurements ffor M estimate
# Monte Carlo Markov Chain Iterations
mcmc_iter = 10000
# Outlier detection quantile threshold
q_MAX = 0.999
# Resampling Period & Starting Time
RS_t0 = pd.Timestamp('2019-08-01T00:00:00')
RS_DT = pd.Timedelta(1,unit='hour')

# Plotting Boolean Switch
isplot = True
issave = True
DPI = 200

#####################################

### DATA MAPPING SECTION ###
DROOT = os.path.join('..','..','..','..')
# Input Data File for AWS (csv format)
IWXSG = os.path.join(DROOT,'data','MINDSatUW','MELT','WXSG','Backup-Y.csv')
# Input Data File for Ablation (csv format)
IABLD = os.path.join(DROOT,'data','MINDSatUW','MELT','LOWER','ABLATION_RATE_LOGS_REDIFFD.csv')
# Output Data File for inversion input data
ORDATA = os.path.join(DROOT,'processed_data','runoff','MTI_evenly_sampled_data.csv')
# Output model parameters and processing parameters(csv format)
OMDATA = os.path.join(DROOT,'processed_data','runoff','model_coefficients_and_parameters.csv')
# Output Summary Data File (csv format)
OSDATA = os.path.join(DROOT,'processed_data','runoff','ETI_Surface_Runoff_Rates.csv')
# Output Summary Data File (csv format)
ORSDATA = os.path.join(DROOT,'processed_data','runoff','ETI_Surface_Runoff_Rates_Resampled.csv')
# Output Root for Figures
ODIR = os.path.join(DROOT,'results','Supplement','Resubmit_1')



#### PHYSICAL MODEL SUBROUTINES ####

### Melt Model Methods ###
def ETI_A_model(X,mf,srf):
    """
    Partial Enhanced Temperature Index Model (Type A, per Gabbi et al. (2014)

    M = mf*X(T) + srf*X(I)

    Function used for inversion for mf and srf using scipy.optimize.curve_fit
    taking the general form of a bi-linear equation:
    Y = aA + bB

    :param X: [2-tuple] with equal-scaled vector arrays:
                X[0]: (T) Temperature values in degC
                X[1]: (I) Incoming Shortwave Radiation in W m-2
    :param mf: [float] melt factor in units mmWE hr-1 degC-1
    :param srf: [float] shortwave radiation factor in units mmWE m2 W-1 hr-1

    :: OUTPUT ::
    :return M: [array-like] modeled meltwater production rates in units mmWE hr-1
    """
    T,I = X
    return mf*T + srf*I


def ETI_A_model_DF(df,mf,srf,Cm,flds={'T':'TEMP(C)','I':'SFLUX(w/m2)'},tunit='hr',T0=0):
    """
    Wrapper to handle data frame indices, calculation of meltwater production variance, and
    inclusion of threshold temperature for complete ETI of the form
        | mf*T + srf*I , T > T0
    M = |      0       , T <= T0
    
    :: INPUTS ::
    :param df: [pandas.DataFrame] indexed data-frame with columns for site temperature (T) 
                    and incoming shortwave radiation (I). These can be aliased using flds (see flds)
    :param mf: [float] estimated melt factor in units mmWE hr-1 degC-1
    :param srf: [float] estimated incoming shortwave radiation factor in units mmWE m2 W-1 hr-1
    :param Cm: [2-by-2 array] model covariance matrix for estimates of mf and srf
    :param flds: [dict] aliases for columns in df that correspond to temperature (T) and SWR (I)
    :param tunit: [str] label to use for time units - gives re-labeling option - default = 'hr'
    :param T0: [float] threshold temperature, default = 0

    :: OUTPUT ::
    :return df_OUT: [pandas.DataFrame] DataFrame with the index from df and the following columns:
                RdotM(mmWE/'tunit'): Meltwater production rate for data pairs (T,I)
                RdotMvar(mmWE2/'tunit'2): Meltwater production rate variance for data pairs (T,I)

    """
    X = (df[flds['T']].values,df[flds['I']].values)
    M = ETI_A_model(X,mf,srf)
    M[M <= T0] = 0
    # Get Melt Rate Variance using formal relationship
    # var_M = T*var_T + I*var_I + 2*T*I*cov_TI
    Mvar = (X[0]**2)*Cm[0,0] + (X[1]**2)*Cm[1,1] + 2.*X[0]*X[1]*Cm[0,1]
    df_OUT = pd.DataFrame({'RdotM(mmWE/%s)'%(tunit):M,'RdotMvar(mmWE2/%s2)'%(tunit):Mvar},index=df.index)

    return df_OUT


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
# Smooth data with rolling average
df_WXS = df_WXR.copy().rolling(SDT).mean()
# Clean up smoothed data index
df_WXS.index -= SDT/2
df_WXS = df_WXS[(df_WXS.index >= df_WXR.index.min())&(df_WXS.index <= df_WXR.index.max())]
# df_WXS = df_WXS*(df_WXR['TEMP(C)']/df_WXR['TEMP(C)'])

### SPECIFY WHICH WEATHER DATAFRAME TO USE IN SUBSEQUENT PROCESSING
df_WX = df_WXS.copy()

### GET RAINFALL RATE
# Delta Index - must be an even value, data are output as window-centered values
I_off = 2
# Get Time Deltas
R_DT = (df_WX.index[I_off:] - df_WX.index[:-1*I_off]).total_seconds()/3600
R_DR = df_WX['RAIN(mm)'].values[I_off:] - df_WX['RAIN(mm)'].values[:-1*I_off]
RdotR = R_DR/R_DT
df_RRATE = pd.DataFrame({'RdotR(mm/hr)':RdotR},index=df_WX.index[int(I_off/2):int(-I_off/2)])


### EXTRACT AVERAGE MELT RATE (M), TEMPERATURE (T), AND INCOMING SHORTWAVE RADIATION (I) VALUES
### FOR ABLATION PAIR TIME WINDOWS
sta = []; Mbar = []; Mvar = []; Mbar_rate = []; Mvar_rate = [];
Tbar = []; Tvar = []; Ibar = []; Ivar = [];
ts = []; te = []; DataFrac = []; DT = []
for s_ in df_ABL['sta'].unique():
    idf_ABL = df_ABL[(df_ABL['sta']==s_) & (df_ABL['tstart'].notna()) & (df_ABL['tend'].notna())]
    nOBS = len(idf_ABL)
    if nOBS > 3:
        # Do all valid pairs of observations
        for i_ in range(nOBS):
            for j_ in range(nOBS):
                # Only assess if i_ start preceds j_ end
                if i_ <= j_:
                    # Get time indices & delta time (in hours)
                    its = idf_ABL['tstart'].iloc[i_]
                    jte = idf_ABL['tend'].iloc[j_]
                    iDT = (jte - its).total_seconds()/3600
                    # Get cumulative ablation for time indices
                    iMbar = idf_ABL['abl_seg_cm'].iloc[i_:j_+1].sum()*10*fWE
                    # Get cumulative ablation variance
                    iMvar = np.sqrt(.5**2 + .5**2)*fWE**2
                    # Get hourly melting rate
                    iMbar_Rate = iMbar/iDT
                    # Get hourly melting rate variance
                    iMvar_Rate = iMvar/iDT
                    # Subsample AWS data
                    WX_IND = (df_WX.index >= its) & (df_WX.index < jte)
                    # Check for gaps in weather data, only use complete records
                    iDataFrac = np.sum(df_WX[WX_IND]['TEMP(C)'].notna())/len(df_WX[WX_IND])
                    # Extract average Temperature (T) and Shortwave Radiation (I)
                    iTbar = df_WX[WX_IND]['TEMP(C)'].mean()
                    iTvar = df_WX[WX_IND]['TEMP(C)'].std()**2
                    iIbar = df_WX[WX_IND]['SFLUX(w/m2)'].mean()
                    iIvar = df_WX[WX_IND]['SFLUX(w/m2)'].std()**2

                    # Compile data
                    sta.append(s_)
                    Mbar.append(iMbar)
                    Mvar.append(iMvar)
                    Mbar_rate.append(iMbar_Rate)
                    Mvar_rate.append(iMvar_Rate)
                    Tbar.append(iTbar)
                    Tvar.append(iTvar)
                    Ibar.append(iIbar)
                    Ivar.append(iIvar)
                    ts.append(its)
                    te.append(jte)
                    DT.append(iDT)
                    DataFrac.append(iDataFrac)
# Format Enhanced Temperature Index Input Data as DataFrame
df_ETI = pd.DataFrame({'sta':sta,'ts':ts,'te':te,'DT(hrs)':DT,\
                       'Mrate(mmWE/hr)':Mbar_rate,'Mbar(mmWE)':Mbar,\
                       'Mrate_var(mmWE2/hr2)':Mvar_rate,'Mvar(mmWE2)':Mvar,\
                       'Tbar(C)':Tbar,'Tvar(C2)':Tvar,\
                       'Ibar(W/m2)':Ibar,'Ivar(W2/m4)':Ivar,\
                       'WX_Coverage':DataFrac})
# Write input data for inversion to disk
df_ETI.to_csv(ORDATA,header=True,index=False)


### CONDUCT INVERSION OF M, T, & I DATA TO ESTIMATE mf AND srf ###
# Apply Filter
M_IND = (df_ETI['WX_Coverage'] >= min_fract) & (df_ETI['DT(hrs)'] >= min_DT)

# Create Data and Uncertainty Inputs
Xd = (df_ETI[M_IND]['Tbar(C)'].values,df_ETI[M_IND]['Ibar(W/m2)'].values)
Xs = (df_ETI[M_IND]['Tvar(C2)'].values,df_ETI[M_IND]['Ivar(W2/m4)'].values)
Yd = df_ETI[M_IND]['Mrate(mmWE/hr)'].values
Ys = df_ETI[M_IND]['Mrate_var(mmWE2/hr2)'].values

# Conduct random-sampling MCMC
# Note: Latin Hypercube Sampler would produce a VERY large perturbation array, 
# just use lots of blind sampling for this MCMC use
MODS = []; COVS = []
print('Conducting MCMC with %d samples'%(mcmc_iter))
for i_ in range(mcmc_iter):
    # Create perturbed Temperature and Solar Radiation Data
    Xrand = (np.random.normal(loc=Xd[0],scale=Xs[0]**0.5),np.random.normal(loc=Xd[1],scale=Xs[1]**0.5))
    # Create perturbed Melt Rate Data
    Yrand = np.random.normal(loc=Yd,scale=Ys**0.5)
    # Do inversion
    Jmod,Jcov = curve_fit(ETI_A_model,Xrand,Yrand,p0=(0.2,0.0004))
    # Append results to holders
    MODS.append(Jmod)
    COVS.append(Jcov)
print('MCMC complete')

# Process Raw Results
MODS = np.array(MODS)
MOD_med_RAW = np.median(MODS,axis=0)
MOD_Q_RAW = np.quantile(MODS,[1 - q_MAX,q_MAX],axis=0)
print('Rejecting outliers beyond quantiles %.4f -- %.4f of both fT and fI'%(1-q_MAX,q_MAX))
# Reject Outliers
O_IDX = (MODS[:,0] >= MOD_Q_RAW[:,0].min()) & (MODS[:,0] <= MOD_Q_RAW[:,0].max()) &\
        (MODS[:,1] >= MOD_Q_RAW[:,1].min()) & (MODS[:,1] <= MOD_Q_RAW[:,1].max())
MODS_filt = MODS[O_IDX,:]
print('Outliers rejected: %d Total samples: %d'%(mcmc_iter - MODS_filt.shape[0],mcmc_iter))
# Calculate Multivariate Gaussian Statistics
Mmod = np.mean(MODS_filt,axis=0)  # Get Expected Model Values
Mcov = np.cov(MODS_filt.T) # Get Model Value Covariance Matrix

# Compile summary of processing in DataFrame
dict_params = {'fT(mmWE/hr-C)':Mmod[0],'fI(mmWE-m2/hr-W)':Mmod[1],\
               'fTvar(mmWE2/hr2-C2)':Mcov[0,0],'fIvar(mmWE2-m4/hr2-W2)':Mcov[1,1],\
               'fTIcov(mmWE2-m2/hr2-C-W)':Mcov[0,1],'mmWE factor':fWE,\
               'WX_Coverage_Min':min_fract,'DT_Est_Min':min_DT,'MCMC_iter':mcmc_iter,\
               'MTI_obs':len(df_ETI),'MTI_kept':len(Yd),\
               'MCMC_kept':MODS_filt.shape[0],'MCMC_quantile_thresh':q_MAX,\
               'TI_smoothing(hrs)':SDT.total_seconds()/3600}
df_params = pd.DataFrame(dict_params,index=[0])
# Write MCMC inversion output to disk
df_params.to_csv(OMDATA,header=True,index=False)


### CONDUCT FORWARD PROBLEM
df_M_HAT = ETI_A_model_DF(df_WX,Mmod[0],Mmod[1],Cm=Mcov,T0=T0)

### COMPILE FORWARD MODEL RESULTS, RAINFALL RATES, AND WX OBSERVATIONS
df_CAT = pd.concat([df_WX[['TEMP(C)','SFLUX(w/m2)']],df_RRATE,df_M_HAT],axis=1,ignore_index=False)
# Stitch small gaps in data
df_OUT = df_CAT.copy().interpolate(limit_direction='both',limit=1)

### WRITE OUTPUT TIMESERIES
df_OUT.to_csv(OSDATA,header=True,index=True)

### WRITE OUTPUT TIMESERIES WITH RESAMPLING
df_RSOUT = pd.concat([pd.DataFrame(index=[RS_t0]),df_OUT.copy()],axis=0,ignore_index=False).resample(RS_DT).mean()
df_RSOUT.index += RS_DT/2
df_RSOUT.to_csv(ORSDATA,header=True,index=True)


#### VISUALIZATION SECTION ####

# Plot Results If Desired
if isplot:
    # Plot ~PDF~ of parameter estimates
    fig,ax = plt.subplots(figsize=(10,10))
    ax.hist2d(MODS_filt[:,0]*24,MODS_filt[:,1]*24,int(mcmc_iter/100))
    ax.plot(Mmod[0]*24,Mmod[1]*24,'xr',markersize=16)
    ax.plot(Mmod[0]*24,Mmod[1]*24,'or',markersize=4)
    
    ax.set_xlabel('Estimated temperature factor [$f_T$]\n($mm w.e. d^{-1} \\degree C^{-1}$)')
    ax.set_ylabel('Estimated incoming shortwave radiation factor [$f_I$]\n($mm w.e. m^2 d^{-1} W^{-1}$)')

    if issave:
        OFILE = os.path.join(ODIR,'FigS6_ETI_MCMC_parameter_estimates_%ddpi.png'%(DPI))
        plt.savefig(OFILE,dpi=DPI,format='png')    

    # # Plot Final Runoff Model
    # plt.figure()
    # plt.plot(df_OUT[['RdotR(mm/hr)','RdotM(mmWE/hr)']].sum(axis=1),'k',label='$\\dot{R}_{surf} (mmWE$ $hr^{-1})$')
    # plt.plot(df_OUT[['RdotR(mm/hr)','RdotM(mmWE/hr)']].sum(axis=1) + df_OUT['RdotMvar(mmWE2/hr2)']**0.5,'k:',label='$s_{\\dot{R}_{surf}} (mmWE$ $hr^{-1})$')
    # plt.plot(df_OUT[['RdotR(mm/hr)','RdotM(mmWE/hr)']].sum(axis=1) - df_OUT['RdotMvar(mmWE2/hr2)']**0.5,'k:')
    # plt.fill_between([df_OUT.index.min(),df_ETI[M_IND]['ts'].min()],[0,0],[7,7],color='yellow',alpha=0.25)
    # plt.fill_between([df_ETI[M_IND]['te'].max(),df_OUT.index.max()],[0,0],[7,7],color='yellow',alpha=0.25)
    # plt.xlim([df_OUT.index.min(),df_OUT.index.max()])
    # plt.ylim([0,7])
    # plt.ylabel('Surface Runoff Production\n [$\\dot{R}_{surf}$] ($mmWE$ $hr^{-1}$)')
    # plt.xlabel('Local Date Time (UTC - 7)')
    plt.show()