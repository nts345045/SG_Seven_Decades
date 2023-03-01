import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def reduce_series(SER,SER24):
	"""
	Remove the rolling average (SER24) from a series of raw data (SER)
	and return a pandas.DataFrame with the residual
	"""
	idf = pd.concat([SER,SER24],axis=1)
	delta = idf[idf.columns[0]].values -idf[idf.columns[1]].values
	df_o = pd.DataFrame(delta,index=idf.index)
	df_o = df_o.rename(columns={0:SER.name})
	return df_o


def fill_between_df(df,fld_u,fld_s,s_fmt='sigma',order=2,fbkwargs={'alpha':0.5}):
    """
    Helper plotting routine
    """
    if s_fmt == 'sigma':
        plt.fill_between(df.index,df[fld_u].values - order*df[fld_s].values,df[fld_u].values + order*df[fld_s].values,**fbkwargs)
    elif s_fmt == 'var':
        plt.fill_between(df.index,df[fld_u].values - order*(df[fld_s].values**0.5),df[fld_u].values + order*(df[fld_s].values**0.5),**fbkwargs)




def rolling_cleanup(proc,src,RW,RU,Clip=0.25):
    """
    Conduct cleaning of dataframe recently operated on by a .rolling() or .resample() method
    to place timestamps at the centers of sampling windows, rather than the starting point
    of sampling windows. Also enforce muting based on data availability from the source dataframe
    and allow a specification of how much of the sampling window to trim off either end of the output
    in order to remove sections affected by sampling edge-effects
    
    :: INPUTS ::
    :type proc: pandas.DataFrame
    :param proc: processed DataFrame from a call of df.rolling() or df.resample()
    :type src: pandas.DataFrame
    :param src: Original DataFrame used to generate proc. Only its index is used
    :type RW: integer
    :param RW: Resampling window length scalar
    :type RU: string
    :param RU: Resampling window unit (must be consistent with period-string fmts)
    :type Clip: fraction in [0,1]
    :param Clip: Fraction of the resampling window length to remove from the end of each
            side of the output data on top of the first and last indexed timestamp from "src"
            
    :: OUTPUT ::
    :rtype out: pandas.DataFrame
    :return out: Cleaned up DataFrame
    """
    # Copy input in case writing to a separate variable
    out = proc.copy()
    # Get "padded" start and end times of input dataframe and the sampling frequency
    tsp = pd.DataFrame([],index=[proc.index[0]])
    tep = pd.DataFrame([],index=[proc.index[-1]])
    freq = proc.index.freq
    # Conduct time-shift to centralize estimate in window
    out.index = out.index - pd.tseries.frequencies.to_offset('%d%s'%(int(RW/2),RU))
    # Trim edges of updated data
    out = out[(out.index >= (src.index[0] + pd.Timedelta(RW*Clip,unit=RU))) & \
              (out.index <= (src.index[-1]- pd.Timedelta(RW*Clip,unit=RU)))]
    # Re-pad to original scale for input "proc"
    out = pd.concat([tsp,out,tep],axis=0)
    # Apply a resampling of equal sampling frequency to re-populate NaN
    out = out.resample(freq).mean()
    return out

def get_df_zscore(df,flds,rwl=10,rwu='min',ofmt='dist'):
    """
    Conduct a rolling() application of a windowed z-score data transform
    on a DataFrame with specified fields and window types

    :: INPUTS ::
    :param df: pandas.DataFrame, DataFrame with an evenly sampled DatetimeIndex
    :param flds: list, list of column names in df to conduct transform on
    :param rwl: int, rOLLING wINDOW lENGTH - scalar value for how long the sampling window is
    :param rwu: str, rOLLING wINDOW uNIT - unit to use when defining window length
    :param ofmt: str, output fmt
            if ofmt == 'dist': Calculate the Euclidian distance of values from flds
            otherwise, return the dataframe contining outputs for each entry in flds

    :: OUTPUT ::
    :return z_score: pandas.DataFrame, contains z-score transforms of each item in flds, unless 
                     ofmt == 'dist', in which case a single-column DataFrame is returned
    """
    if rwl % 2 != 0:
        rwl = int(rwl/2)*2
        print('Correcting rwl to nearest base-2 integer')
    tmp_u = df[flds].rolling('%d%s'%(rwl,rwu)).mean()
    # tmp_u = rolling_cleanup(df,tmp_u,'%d%s'%(rwl/2,rwu),inplace=inplace)
    tmp_u = rolling_cleanup(tmp_u,df,rwl,rwu,Clip=0.0)
    tmp_o = df[flds].rolling('%d%s'%(rwl,rwu)).std()
    # tmp_o = rolling_cleanup(df,tmp_o,'%d%s'%(rwl/2,rwu),inplace=inplace)
    tmp_o = rolling_cleanup(tmp_o,df,rwl,rwu,Clip=0.0)
    z_score = (df[flds] - tmp_u)/tmp_o
    if ofmt == 'dist':
        z_score = ((z_score**2).sum(axis=1))**0.5
    return z_score

def roll_wgtwin(df,min_pct=0.05,rwl=20,rwu='min',cols=[('mX','vxx'),('mY','vyy'),('mZ','vzz')]):
    df_OUT = pd.DataFrame()
    for i_ in range(len(df)):
        DTr = df.index[i_]
        print('%s -- %s'%(DTr,df.index.max()))
        idf = df[np.abs((df.index - DTr).total_seconds()) <= pd.Timedelta(rwl/2,unit=rwu).total_seconds()]
        tmp = {}
        for u_,v_ in cols:
            dx = (idf.index - idf.index[0]).total_seconds()
            dy = idf[u_].values
            wy = idf[v_].values**-1
            idx = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(wy)
            if len(idf) > 3 and sum(idx)/len(idf) >= min_pct:
                try:
                    m,c = np.polyfit(dx[idx],dy[idx],1,w=wy[idx],cov='unscaled')
                    
                except ValueError:
                    m = [np.nan]; c = [[np.nan, np.nan], [np.nan, np.nan]]
            else:
                m = [np.nan]; c = [[np.nan, np.nan], [np.nan, np.nan]]

            tmp.update({u_:m[0],v_+'mm':c[0][0],v_+'tt':c[1][1],v_+'mt':c[0][1]})

        odf = pd.DataFrame(tmp,index=[DTr])
        df_OUT = pd.concat([df_OUT,odf],axis=0,ignore_index=False)

    return df_OUT



def winslope(arg,min_pct=0.05):
    """
    Take an input Series argument and return the 1st order linear fit slope
    Used as a wrapper in pandas resampler *.apply() calls to do first derivative
    estimates for rolling windows and resampling

    :: INPUTS ::
    :type arg: pandas.Series
    :param arg: Series with a DatetimeIndex
    :type min_pct: float
    :param min_pct: minimum (fractional) percent of the data that are
        required to be finite values in order to return a non NaN.

    :: OUTPUT ::
    :rtype: float or NaN
    :return: Slope of line, if calculable

    """
    dt = (arg.index - arg.index[0]).total_seconds()
    dy = arg.values
    idx = np.isfinite(dt) & np.isfinite(dy)
    if len(arg) > 3 and sum(idx)/len(arg) >= min_pct:
        try:
            m = np.polyfit(dt[idx],dy[idx],1)
            return m[0]
        except ValueError:
            return np.nan
    else:
        return np.nan

def wgtwinslope(arg,wgts=None,cov=True,min_pct=0.05):
    """
    wgtwinslope - Return slope of a collection of points with specified 1/var weights using
    the numpy.polyfit() method.

     (see numpy.polyfit)
    """
    dt = (arg.index - arg.index[0]).total_seconds()
    dy = arg.values
    if wgts is not None:
        idx = np.isfinite(dt) & np.isfinite(dy) & np.isfinite(wgts)
    else:
        idx = np.isfinite(dt) & np.isfinite(dy)
    if len(arg) > 3 and sum(idx)/len(arg) >= min_pct:
        if wgts is not None:
            try:
                m = np.polyfit(dt[idx],dy[idx],1,w=1/wgts[idx],cov=cov)
                return m[0]
            except ValueError:
                return np.nan
        elif wgts is None:
            try:
                m = np.polyfit(dt[idx],dy[idx],1)
                return m[0]
            except ValueError:
                return np.nan
        else:
            return np.nan
    else:
        return np.nan


def wgtwinslopevar(arg,wgts=None):
  if len(arg) > 3:
      dt = (arg.index - arg.index[0]).total_seconds()
      dy = arg.values
        
      if wgts is not None:
          wgt = wgts[arg.index].values
          idx = np.isfinite(dt) & np.isfinite(dy) & np.isfinite(wgt)
          m,v = np.polyfit(dt[idx],dy[idx],1,w=1/wgt[idx],cov=True)
      else:
          idx = np.isfinite(dt) & np.isfinite(dy)
          m,v = np.polyfit(dt[idx],dy[idx],1,cov=True)
      return v[0][0]
  else:
      return np.nan   


def model_velocities_v2(proc,src,RW,RU,wskw={'min_pct':0.1},Clip=0.25,PolyOpt=True,PolyMaxOrder=4,verb=True):
    # Get each field to model velocity from "process" (proc)
    if isinstance(proc,pd.DataFrame):
        flds = proc.columns
    elif isinstance(proc,pd.Series):
        flds = proc.name

    ### Impose Polyfit Clause: In the case where Poly=True or Poly=None
    # Use a polynomial fit to the data when resampling window comes sufficiently close to 
    # Get first and last datapoints with non-NaN entries from the proc datasource
    nanind = proc[flds[0]].isna()
    cdf = proc[~nanind]
    its = cdf.index.min()
    ite = cdf.index.max()
    idt = (ite - its).total_seconds()
    iwl = pd.Timedelta(RW,unit=RU).total_seconds()

    df_out = pd.DataFrame([],index=proc.index)

    # If PolyOpt is True and window length is G.E. 1/4 the full data length, do a polynomial fit for speed-up
    if int(np.ceil(idt/iwl)) <= PolyMaxOrder and PolyOpt is True:
        deg = int(np.ceil(idt/iwl))
        for i_,fld in enumerate(flds):
            # Convert index to elapsed seconds
            tstar = (proc.index - proc.index.min()).total_seconds()
            idata = proc[fld].values
            # Sift out non-valid entries
            idx = np.isfinite(idata)
            # Do polynomial fit with time in the X and values in the Y
            ifit = np.polyfit(tstar[idx],idata[idx],deg)
            # Take Derivative
            ider = np.polyder(ifit,m=1)
            # Model Curve at tstar points
            idfun = np.poly1d(ider)
            # Create modelled values for 
            idmod = idfun(tstar)
            # Compose into full dataframe
            d_vi = pd.DataFrame(idmod,index=proc.index,columns=[fld])
            df_out = pd.concat([df_out,df_vi],axis=1)
    else:
        for i_,fld in enumerate(flds):
            if verb:
                print('Processing %s (%d/%d)'%(fld,i_+1,len(flds)))
            ## Conduct Velocity Modelling
            s_vi = proc[fld].copy().rolling('%d%s'%(RW,RU)).apply(winslope,kwargs=wskw,raw=False)
            # Do Cleanup
            d_vi = pd.DataFrame(s_vi.values,index=s_vi.index,columns=[fld])
            d_vi = rolling_cleanup(d_vi,src,RW,RU,Clip=Clip)
    #         d_vi = d_vi.rename(columns={fld:fld+'dt'})
            # if verb:
            #     print('Processed, cleaned, renamed: %s'%(d_vi.columns[0]))
            if verb:
                print('Processed & Cleaned Up')
            df_out = pd.concat([df_out,d_vi],axis=1)
    return df_out


# Application of moving-window WLS to indexed DataFrames

def detect_dt(index, method=np.median,already_sorted=True,fmt='Timedelta'):
    """
    Utility method to estimate the sampling rate of a DatetimeIndex
    This has an in-line sort
    :: INPUTS ::
    :type index: pandas.DatetimeIndex
    :param index: index to assess
    :type method: method
    :param method: method to apply to a np.array of finite-differenced index values in seconds
    :type already_sorted: bool
    :param already_sorted: Default True, otherwise, apply .sort_values() to index before differencing
    :type fmt: str
    :param fmt: output fmt. Currently accepts: 'Timedelta' for pd.Timedelta class output (unit='sec')
                                                 and 'float' for numpy.float class output (in seconds)

    :: OUTPUT ::
    :return dt_hat: value output by 'method' in 'fmt' specified class
    """
    if not already_sorted:
        iidx = index.copy().sort_values()
    else:
        iidx = index.copy()
    dtv = (iidx[1:] - iidx[:-1]).total_seconds()
    dt_hat = method(dtv)
    if fmt == 'Timedelta':
        return pd.Timedelta(dt_hat,unit='sec')
    elif fmt == 'float':
        return dt_hat


def detect_dx(index, method=np.median,already_sorted=True,method_kwargs={}):
    """
    Utility method to estimate the sample spacing of a FloatIndex
    This has an in-line sort
    :: INPUTS ::
    :type index: pandas.Index
    :param index: index to assess
    :type method: method
    :param method: method to apply to a np.array of finite-differenced index values in seconds
    :type already_sorted: bool
    :param already_sorted: Default True, otherwise, apply .sort_values() to index before differencing

    :: OUTPUT ::
    :return dt_hat: value output by 'method'
    """
    if not already_sorted:
        iidx = index.copy().sort_values()
    else:
        iidx = index.copy()
    dxv = iidx[1:] - iidx[:-1]
    dx_hat = method(dxv)
    return dx_hat


def apply_windowed_method(Series,wl,ws,method=np.nanmedian,wpos='center',method_kwargs={}):
    """
    Apply some method that accepts an argument of a numpy.ndarray of shape n, or n,1
    and returns a single value. This works around some of the bugs in the Pandas
    rolling() & resample() sub-methods for applying such operators

    This results in 2 modifications:
    1) application of 'method' to the input data
    2) uniform sampling with 'ws' sampling interval

    :: INPUTS ::
    :type Series: pandas.Series
    :param Series: Series with a monotonically increasing index of either pandas.DatetimeIndex or float-like values
    :type wl: pandas.Timedelta or float, must be compatable with the index of Series
    :param wl: window length
    :type ws: pandas.Timedelta or float, must be compatable with the index of Series
    :param ws: window step-size and new sampling period for the output series
    :type method: python method (default is numpy.nanmedian)
    :param method: callable python method to apply to the Series
    :type wpos: str
    :param wpos: indexing position relative to window. Options are 'left', 'center', or 'right'


    """
    #Create Output DataFrame
    df_OUT = pd.DataFrame()
    
    ### Construct Sampling Scheme
    # Detect Index Type & Median Spacing
    if isinstance(Series.index.min(),pd.Timestamp):
        dx = detect_dt(Series.index)
        DTIDX = True
    else:
        dx = detect_dx(Series.index)
        DTIDX = False

    if dx > ws:
        ws = dx
        print('WARNING: Window step smaller than median sample-size!')
        print('---> Continuing with median sample-size as ws: %s'%(dx))

    # Estimate the number of windows
    Io = Series.index.min()
    If = Series.index.max()
    # Calculate the maximum number of possible windows, allowing for off-end extension by wl*(1 - min_frac)
    nwinds = (If - Io)/ws
    # Get the expected number of samples per window
    ntotal = np.floor(wl/dx)

    # Get window-center relative position in wl
    if wpos.lower() == 'left':
        cw_ = -0.5
    elif wpos.lower() == 'center':
        cw_ = 0.0
    elif wpos.lower() == 'right':
        cw_ = 0.5
    # Create Window Index (WINDEX) that assumes centered indexing
    WINDEX = Io + 0.5*wl + np.arange(nwinds)*ws

    # Create holder lists for new index and derivative data
    w_ind = []; w_data = []
    for w_val in tqdm(WINDEX):
        w_s = w_val - (0.5 + cw_)*wl # Starting Index
        w_e = w_val + (0.5 + cw_)*wl # Ending Index
        w_ref = w_val + cw_*wl # Reference Index
        # Fetch Sub-DataFrame view
        IDX = (Series.index >= w_s) & (Series.index < w_e)
        # Fetch & Subsample DataFrame
        iS = Series[IDX].copy()


        ### APPLY METHOD ########################
        ivalue = method(iS,**method_kwargs)
        #########################################

        w_ind.append(w_ref)
        w_data.append(ivalue)
    # Create output Series
    S_out = pd.Series(w_data,index=w_ind,name=Series.name)
    return S_out





def lin_fun(x,m,b):
    return m*x + b


def rolling_linear_WLS(df,flds={'x':('mX','vxx'),'y':('mY','vyy')},wl=pd.Timedelta(6,unit='hour'),ws=pd.Timedelta(1,unit='min'),min_frac=0.75,wpos='center',tmp_save_pre='tmp_rlWLS_'):
    """
    Conduct a rolling/stepping window application of a Weighted Least Squares 1st order fitting to 
    indexed values and their variances from DataFrame using the scipy.optimize.curve_fit method, 
    returning an indexed DataFrame with model estimtes (slope & intercept) and their (co)variances. 

    Uses the following subroutine methods:
    detect_dt - find the median sampling period of an input pandas.DatetimeIndex, return in pandas.Timedelta format
    detect_dx - find the median sampling period of an input float-valued index, return in float
    lin_fun - linear equation to pass to curve_fit


    :: INPUTS ::
    :type df: pandas.DataFrame
    :param df: DataFrame with a monotonically increasing index
    :type flds: dict of 2-tuples
    :param flds: Dictionary of associated (variable,variance) fields on which to operate
                This assumes that values in the variance field are std**2, and thus weights
                for numpy.polyfit are w = variance**(-0.5).
    :type wl: float or pandas.Timedelta
    :param wl: length of window for sampling df
    :type ws: None, float, or pandas.Timedelta
    :param ws: size of step to take for successive windows
    :type min_frac: float
    :param min_frac: minimum fraction of window containing valid data (assumes a uniform sampling period for data)
                    candidate windows with too-little data are returned as an indexed line of NaNs
    :type wpos: str
    :param wpos: position to index window, regardless of data content, and index value to reference inputs to 
                numpy.polyfit for it's independent variable argument (1st position)
                Accepted Arguments:
                    'left': index outputs of WLS to left edge of window
                    'right': index outputs of WLS to right edge of window
                    'center': index outputs of WLS to the center of window
    :type tmp_save_pre: str
    :param tmp_save_pre: location and start to file-name for saving individual flds tuple results. Save path-name
                results in: '%s_%s.csv'%(tmp_save_pre,k_), with k_ the k**th key in flds
    :: OUTPUT ::
    :rtype df_out: pandas.DataFrame
    :return df_out: output data with columns: $key_m, $key_b, $key_vmm, $key_vbb, $key_vmb
                    for each dictionary key-term provided in flds. E.g, x_m, ... , x_vmb.
                    $key_m : best-fit slope
                    $key_b : best-fit intercept (referenced to indexed value)
                    $key_vmm: best-fit slope variance
                    $key_vbb: best-fit intercept variance
                    $key_vmb: best-fit slope/intercept covariance

    """
    # Create Output DataFrame
    df_OUT = pd.DataFrame()
    
    ### Construct Sampling Scheme
    # Detect Index Type & Median Spacing
    if isinstance(df.index.min(),pd.Timestamp):
        dx = detect_dt(df.index)
        DTIDX = True
    else:
        dx = detect_dx(df.index)
        DTIDX = False

    if dx > ws:
        ws = dx
        print('WARNING: Window step smaller than median sample-size!')
        print('---> Continuing with median sample-size as ws: %s'%(dx))

    # Estimate the number of windows
    Io = df.index.min()
    If = df.index.max()
    # Calculate the maximum number of possible windows, allowing for off-end extension by wl*(1 - min_frac)
    nwinds = (If - Io + (1-min_frac)*2*wl)/ws
    # Get the expected number of samples per window
    ntotal = np.floor(wl/dx)

    # Get window-center relative position in wl
    if wpos.lower() == 'left':
        cw_ = -0.5
    elif wpos.lower() == 'center':
        cw_ = 0.0
    elif wpos.lower() == 'right':
        cw_ = 0.5
    # Create Window Index (WINDEX) that assumes centered indexing
    WINDEX = Io + (min_frac - 0.5)*wl + np.arange(nwinds)*ws

    ## OUTER LOOP ##
    for k_ in list(flds.keys()):
        print('Starting to process %s (field %s, variance %s)'%(k_,flds[k_][0],flds[k_][1]))
        # Create temporary lists to hold outputs
        k_ind = []; k_m = []; k_b = []; k_vmm = []; k_vbb = []; k_vmb = []; k_frac = [];
    
        ## INNER LOOP ##

        for w_val in tqdm(WINDEX):
            w_s = w_val - (0.5 + cw_)*wl # Starting Index
            w_e = w_val + (0.5 + cw_)*wl # Ending Index
            w_ref = w_val + cw_*wl # Reference Index
            # Fetch Sub-DataFrame view
            IDX = (df.index >= w_s) & (df.index < w_e)
            # Fetch & Subsample DataFrame
            idf = df[IDX].copy()
            # Get series
            iS_u = idf[flds[k_][0]]
            iS_v = idf[flds[k_][1]]
            # Get fraction of valid entries
            ind = np.isfinite(iS_u.values) & np.isfinite(iS_v.values) & np.isfinite(iS_u.index)
            nvalid = int(np.sum(ind))
            valid_frac = nvalid/ntotal
            # Document indices & valid fraction for output
            k_ind.append(w_ref)
            k_frac.append(valid_frac)
            # If the minimum data requirements are met
            if valid_frac >= min_frac:
                # Get independent variable values
                if DTIDX:
                    dx = (iS_u.index[ind] - w_ref).total_seconds()
                elif not DTIDX:
                    dx = (iS_u.index[ind] - w_ref)
                # Get dependent variable values
                dy = iS_u.values[ind]
                # Get dependent variable variances
                dv = iS_v.values[ind]

                ####### Core Process ##############################
                p0 = 0,np.mean(dy)
                mod,cov = curve_fit(lin_fun,dx,dy,p0,sigma=dv**0.5)
                ###################################################

                # mod,cov = np.polyfit(dx,dy,1,w=dv**(-1),cov='unscaled')
                # Save Results
                k_m.append(mod[0]); k_b.append(mod[1]); k_vmm.append(cov[0][0]); k_vbb.append(cov[1][1]); k_vmb.append(cov[0][1])
            # Write nan entries if insufficient data are present
            else:
                k_m.append(np.nan); k_b.append(np.nan); k_vmm.append(np.nan); k_vbb.append(np.nan); k_vmb.append(np.nan)
        df_k = pd.DataFrame({k_+'_m':k_m,k_+'_b':k_b,k_+'_frac':k_frac,\
                             k_+'_vmm':k_vmm,k_+'_vbb':k_vbb,k_+'_vmb':k_vmb},\
                             index=k_ind)
        # Save result
        save_arg = '%s_%s.csv'%(tmp_save_pre,k_)
        print("Processing of %s complete. Saving as: %s"%(k_,save_arg))
        df_k.to_csv(save_arg, header=True, index=True)
        # Concatenate multiple field pairs into a final output
        print('Concatenating results to main output')
        df_OUT = pd.concat([df_OUT,df_k],ignore_index=False,axis=1)
    return df_OUT



def rolling_linear_LS(df,flds={'x':'mX'},wl=300,ws=10,min_frac=0.5,wpos='center',tmp_save_pre='tmp_rlWLS_'):
    """
    Conduct a rolling/stepping window application of a Weighted Least Squares 1st order fitting to 
    indexed values and their variances from DataFrame using the scipy.optimize.curve_fit method, 
    returning an indexed DataFrame with model estimtes (slope & intercept) and their (co)variances. 

    Uses the following subroutine methods:
    detect_dt - find the median sampling period of an input pandas.DatetimeIndex, return in pandas.Timedelta format
    detect_dx - find the median sampling period of an input float-valued index, return in float
    lin_fun - linear equation to pass to curve_fit


    :: INPUTS ::
    :type df: pandas.DataFrame
    :param df: DataFrame with a monotonically increasing index
    :type flds: dict of 2-tuples
    :param flds: Dictionary of associated (variable,variance) fields on which to operate
                This assumes that values in the variance field are std**2, and thus weights
                for numpy.polyfit are w = variance**(-0.5).
    :type wl: float or pandas.Timedelta
    :param wl: length of window for sampling df
    :type ws: None, float, or pandas.Timedelta
    :param ws: size of step to take for successive windows
    :type min_frac: float
    :param min_frac: minimum fraction of window containing valid data (assumes a uniform sampling period for data)
                    candidate windows with too-little data are returned as an indexed line of NaNs
    :type wpos: str
    :param wpos: position to index window, regardless of data content, and index value to reference inputs to 
                numpy.polyfit for it's independent variable argument (1st position)
                Accepted Arguments:
                    'left': index outputs of WLS to left edge of window
                    'right': index outputs of WLS to right edge of window
                    'center': index outputs of WLS to the center of window
    :type tmp_save_pre: str
    :param tmp_save_pre: location and start to file-name for saving individual flds tuple results. Save path-name
                results in: '%s_%s.csv'%(tmp_save_pre,k_), with k_ the k**th key in flds
    :: OUTPUT ::
    :rtype df_out: pandas.DataFrame
    :return df_out: output data with columns: $key_m, $key_b, $key_vmm, $key_vbb, $key_vmb
                    for each dictionary key-term provided in flds. E.g, x_m, ... , x_vmb.
                    $key_m : best-fit slope
                    $key_b : best-fit intercept (referenced to indexed value)
                    $key_vmm: best-fit slope variance
                    $key_vbb: best-fit intercept variance
                    $key_vmb: best-fit slope/intercept covariance

    """
    # Create Output DataFrame
    df_OUT = pd.DataFrame()
    
    ### Construct Sampling Scheme
    # Detect Index Type & Median Spacing
    if isinstance(df.index.min(),pd.Timestamp):
        dx = detect_dt(df.index)
        DTIDX = True
    else:
        dx = detect_dx(df.index)
        DTIDX = False

    if dx > ws:
        ws = dx
        print('WARNING: Window step smaller than median sample-size!')
        print('---> Continuing with median sample-size as ws: %s'%(dx))

    # Estimate the number of windows
    Io = df.index.min()
    If = df.index.max()
    # Calculate the maximum number of possible windows, allowing for off-end extension by wl*(1 - min_frac)
    nwinds = (If - Io + (1-min_frac)*2*wl)/ws
    # Get the expected number of samples per window
    ntotal = np.floor(wl/dx)

    # Get window-center relative position in wl
    if wpos.lower() == 'left':
        cw_ = -0.5
    elif wpos.lower() == 'center':
        cw_ = 0.0
    elif wpos.lower() == 'right':
        cw_ = 0.5
    # Create Window Index (WINDEX) that assumes centered indexing
    WINDEX = Io + (min_frac - 0.5)*wl + np.arange(nwinds)*ws

    ## OUTER LOOP ##
    for k_ in list(flds.keys()):
        # print('Starting to process %s (field %s, variance %s)'%(k_,flds[k_][0],flds[k_][1]))
        # Create temporary lists to hold outputs
        k_ind = []; k_m = []; k_b = []; k_vmm = []; k_vbb = []; k_vmb = []; k_frac = [];
    
        ## INNER LOOP ##

        for w_val in tqdm(WINDEX):
            w_s = w_val - (0.5 + cw_)*wl # Starting Index
            w_e = w_val + (0.5 + cw_)*wl # Ending Index
            w_ref = w_val + cw_*wl # Reference Index
            # Fetch Sub-DataFrame view
            IDX = (df.index >= w_s) & (df.index < w_e)
            # Fetch & Subsample DataFrame
            idf = df[IDX].copy()
            # Get series
            iS_u = idf[flds[k_]]
            # iS_v = idf[flds[k_][1]]
            # Get fraction of valid entries
            ind = np.isfinite(iS_u.values) & np.isfinite(iS_u.index) # & np.isfinite(iS_v.values)
            nvalid = int(np.sum(ind))
            valid_frac = nvalid/ntotal
            # Document indices & valid fraction for output
            k_ind.append(w_ref)
            k_frac.append(valid_frac)
            # If the minimum data requirements are met
            if valid_frac >= min_frac:
                # Get independent variable values
                if DTIDX:
                    dx = (iS_u.index[ind] - w_ref).total_seconds()
                elif not DTIDX:
                    dx = (iS_u.index[ind] - w_ref)
                # Get dependent variable values
                dy = iS_u.values[ind]
                # Get dependent variable variances
                # dv = iS_v.values[ind]

                ####### Core Process ##############################
                p0 = 0,np.mean(dy)
                mod,cov = curve_fit(lin_fun,dx,dy,p0)#,sigma=dv**0.5)
                ###################################################

                # mod,cov = np.polyfit(dx,dy,1,w=dv**(-1),cov='unscaled')
                # Save Results
                k_m.append(mod[0]); k_b.append(mod[1]); k_vmm.append(cov[0][0]); k_vbb.append(cov[1][1]); k_vmb.append(cov[0][1])
            # Write nan entries if insufficient data are present
            else:
                k_m.append(np.nan); k_b.append(np.nan); k_vmm.append(np.nan); k_vbb.append(np.nan); k_vmb.append(np.nan)
        df_k = pd.DataFrame({k_+'_m':k_m,k_+'_b':k_b,k_+'_frac':k_frac,\
                             k_+'_vmm':k_vmm,k_+'_vbb':k_vbb,k_+'_vmb':k_vmb},\
                             index=k_ind)
        # Save result
        save_arg = '%s_%s.csv'%(tmp_save_pre,k_)
        print("Processing of %s complete. Saving as: %s"%(k_,save_arg))
        df_k.to_csv(save_arg, header=True, index=True)
        # Concatenate multiple field pairs into a final output
        print('Concatenating results to main output')
        df_OUT = pd.concat([df_OUT,df_k],ignore_index=False,axis=1)
    return df_OUT