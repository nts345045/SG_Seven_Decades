import numpy as np
import pandas as pd


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
    :param RU: Resampling window unit (must be consistent with period-string formats)
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
    :param ofmt: str, output format
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



    

# def wgtwinslope(arg,wgts=None,min_pct=0.05):
#     dt = (arg.index - arg.index[0]).total_seconds()
#     dy = arg.values
#     if wgts is not None:
#         idx = np.isfinite(dt) & np.isfinite(dy) & np.isfinite(wgts)
#     else:
#         idx = np.isfinite(dt) & np.isfinite(dy)
#     if len(arg) > 3 and sum(idx)/len(arg) >= min_pct:
#         if wgts is not None:
#             try:
#                 m = np.polyfit(dt[idx],dy[idx],1,w=1/wgts[idx])
#                 return m[0]
#             except ValueError:
#                 return np.nan
#         elif wgts is None:
#             try:
#                 m = np.polyfit(dt[idx],dy[idx],1)
#                 return m[0]
#             except ValueError:
#                 return np.nan
#         else:
#             return np.nan
#     else:
#         return np.nan


# def wgtwinslopevar(arg,wgts=None):
# 	if len(arg) > 3:
# 		dt = (arg.index - arg.index[0]).total_seconds()
# 		dy = arg.values
		
# 		if wgts is not None:
# 			wgt = wgts[arg.index].values
# 			idx = np.isfinite(dt) & np.isfinite(dy) & np.isfinite(wgt)
# 			m,v = np.polyfit(dt[idx],dy[idx],1,w=1/wgt[idx],cov=True)
# 		else:
# 			idx = np.isfinite(dt) & np.isfinite(dy)
# 			m,v = np.polyfit(dt[idx],dy[idx],1,cov=True)
# 		return v[0][0]
# 	else:
# 		return np.nan		

# def winslopevar(arg):
# 	if len(arg) > 3:
# 		dt = (arg.index - arg.index[0]).total_seconds()
# 		dy = arg.values
# 		idx = np.isfinite(dt) & np.isfinite(dy)
# 		m,v = np.polyfit(dt[idx],dy[idx],1,cov=True)
# 		return v[0][0]
# 	else:
# 		return np.nan

# def wincurve(arg):
#     # Extract 0.5*avg.acceleration with a 2nd order polynomial fit
#     if len(arg) > 3:
#         dt = (arg.index - arg.index[0]).total_seconds()
#         dy = arg.values
#         m = np.polyfit(dt,dy,2)
#         return m[0]
#     else:
#         return np.nan

# def winprint(arg):
#     print(arg)
#     print(type(arg))
#     return(np.nan)





# def model_velocities(proc,src,RW,RU,wskw={'min_pct':0.1},Clip=0.25,PolyOpt=True,PolyMaxOrder=4,verb=True):
#     # Get each field to model velocity from "process" (proc)
#     if isinstance(proc,pd.DataFrame):
#         flds = proc.columns
#     elif isinstance(proc,pd.Series):
#         flds = proc.name
#     df_out = pd.DataFrame([],index=proc.index)
#     for i_,fld in enumerate(flds):
#         if verb:
#             print('Processing %s (%d/%d)'%(fld,i_+1,len(flds)))
#         # Conduct Velocity Modelling
#         s_vi = proc[fld].copy().rolling('%d%s'%(RW,RU)).apply(winslope,kwargs=wskw,raw=False)
#         # Do Cleanup
#         d_vi = pd.DataFrame(s_vi.values,index=s_vi.index,columns=[fld])
#         d_vi = rolling_cleanup(d_vi,src,RW,RU,Clip=Clip)
# #         d_vi = d_vi.rename(columns={fld:fld+'dt'})
#         # if verb:
#         #     print('Processed, cleaned, renamed: %s'%(d_vi.columns[0]))
#         if verb:
#             print('Processed & Cleaned Up')
#         df_out = pd.concat([df_out,d_vi],axis=1)
#     return df_out


