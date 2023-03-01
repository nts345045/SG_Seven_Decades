import pandas as pd
import numpy as np
from pyproj import Proj
from glob import glob
from tqdm import tqdm # Provides progress bars
import os
import sys

### STEP1 Supporting Routines ###

def global2local(lons,lats,elevs,lon0,lat0,elev0,crfun=None):
	"""
	Convert longitude, latitude, and elevation vectors (of equal length) into
	localized UTM coordinates using a reference lat/lon/elevation

	:: INPUTS ::
	:param lons: array-like, longitudes
	:param lats: array-like, latitudes
	:param elevs: array-like, elevations
	:param lon0: float, reference longitude
	:param lat0: float, reference latitude
	:param elev0: float, reference elevation
	:param crfun: None OR string, cartesian reference function string, formatted for pyproj.Proj()

	:: OUTPUT ::
	:return XYZ_vect: n x 3 array of X,Y,Z position data
	"""
	if crfun is None:
		myProj = Proj("+proj=utm +zone=11U, +south +ellipse=WGS84 +datum=WGS84 +units=m +no_defs")
	else:
		myProj = Proj(crfun)

	UTMx,UTMy = myProj(lons,lats)
	x0,y0 = myProj(lon0,lat0)
	xl = UTMx - x0
	yl = UTMy - y0
	zl = elevs - elev0
	return np.c_[xl,yl,zl]


def model_gaps(data_df,gap_df,param_wind=pd.Timedelta(1,'h'),method='constant_acceleration',\
			   cxyz=['mE','mN','mZ'],oxyz=['sXX','sYY','sZZ']):
	"""
	:: INPUT ::
	:param data_df: DataFrame containing meter-scaled location values (fields defined in xyz_cols)
	:param gap_df: DataFrame containing gap-start, gap-stop, and True/False if processing of gap should occur
	:param param_wind_sec: float value [sec] for estimating velocity pre- and post-gap
	:param method: gap displacement modelling method to incorporate:
			'leading_velocity': offset calculated from last pre-gap location 
								and the WLS-estimate of mean velocity pre-gap
			'lagging_velocity': offset calculated from last pre-gap location
								and the WLS-estimate of mean velocity post-gap
			'average_velocity': offset calculated from last pre-gap location
								and the average of the WLS-estimates of mean velocity
								pre- and post-gap
			'constant_acceleration': offset calculated from the last pre-gap location and a 
								2nd order fit to the pre- and post-gap WLS-estimates of
								mean velocity.

	:: OUTPUT ::
	:return df: DataFrame containing processed data
	:return df_mod: DataFrame containing model parameters for each gap
	"""
	# Create a subset copy of the input dataframe
	df = data_df.copy()
	# Create holder for model parameters
	mods = []
	# Iterate across gaps defined in gap_df
	for i_ in range(len(gap_df)):
		print("Processing {} to {} (Model: {})".format(gap_df.iloc[i_]['gstart'],gap_df.iloc[i_]['gstop'],gap_df.iloc[i_]['model']))
		# Define indices to sample from data to get velocities
		pre_win = [gap_df['gstart'].values[i_] - param_wind, gap_df['gstart'].values[i_]]
		pos_win = [gap_df['gstop'].values[i_],   gap_df['gstop'].values[i_] + param_wind]

		# Trim out data from between gaps
		df = df[(df.index < pre_win[1]) | (df.index > pos_win[0])]
		
		# If Processing, Proceed
		if gap_df['model'].values[i_]:
			# Get second-length of data gap
			delta = pd.Timedelta(pos_win[0] - pre_win[1]).total_seconds()
			# Get indices of post-gap data
			pind = df[df.index >= pos_win[0]].index
			# Get pre-gap and post-gap data
			dfi = df[(df.index >= pre_win[0]) & (df.index <= pre_win[1])]
			dfo = df[(df.index >= pos_win[0]) & (df.index <= pos_win[1])]

			# Estimate pre-gap mean velocity, if applicable, with weighted least-squares fit
			if method in ['leading_velocity','mean_velocity']:
				m1x = np.polyfit(dfi.index.values.astype('float')*1e-9,dfi[cxyz[0]].values,1,\
								 w=dfi[oxyz[0]].values**-2)
				m1y = np.polyfit(dfi.index.values.astype('float')*1e-9,dfi[cxyz[1]].values,1,\
								 w=dfi[oxyz[1]].values**-2)
				m1z = np.polyfit(dfi.index.values.astype('float')*1e-9,dfi[cxyz[2]].values,1,\
								 w=dfi[oxyz[2]].values**-2)

			# Estimate post-gap mean velocity, if applicable, with weighted least-squares fit
			if method in ['lagging_velocity','mean_velocity']:
				m2x = np.polyfit(dfo.index.values.astype('float')*1e-9,dfo[cxyz[0]].values,1,\
								 w=dfo[oxyz[0]].values**-2)
				m2y = np.polyfit(dfo.index.values.astype('float')*1e-9,dfo[cxyz[1]].values,1,\
								 w=dfo[oxyz[1]].values**-2)
				m2z = np.polyfit(dfo.index.values.astype('float')*1e-9,dfo[cxyz[2]].values,1,\
								 w=dfo[oxyz[2]].values**-2)

			if method == 'leading_velocity':
				mdx = m1x[0]*delta
				mdy = m1y[0]*delta
				mdz = m1z[0]*delta
				modx = [m1x[0], df[df.index <= pre_win[1]][cxyz[0]].values[-1]]
				mody = [m1y[0], df[df.index <= pre_win[1]][cxyz[1]].values[-1]]
				modz = [m1z[0], df[df.index <= pre_win[1]][cxyz[2]].values[-1]]

			elif method == 'lagging_velocity':
				mdx = m2x[0]*delta
				mdy = m2y[0]*delta
				mdz = m2z[0]*delta
				modx = [m2x[0], df[df.index <= pre_win[1]][cxyz[0]].values[-1]]
				mody = [m2y[0], df[df.index <= pre_win[1]][cxyz[1]].values[-1]]
				modz = [m2z[0], df[df.index <= pre_win[1]][cxyz[2]].values[-1]]

			elif method == 'mean_velocity':
				mdx = ((m2x[0] + m1x[0])/2.)*delta
				mdy = ((m2y[0] + m1y[0])/2.)*delta
				mdz = ((m2z[0] + m1z[0])/2.)*delta
				modx = [((m1x[0] + m2x[0])/2.), df[df.index <= pre_win[1]][cxyz[0]].values[-1]]
				mody = [((m1x[0] + m2x[0])/2.), df[df.index <= pre_win[1]][cxyz[1]].values[-1]]
				modz = [((m1x[0] + m2x[0])/2.), df[df.index <= pre_win[1]][cxyz[2]].values[-1]]

			# Calculate Offsets
			dx =   (df[df.index <= pre_win[1]][cxyz[0]].values[-1] - \
					df[df.index >= pos_win[0]][cxyz[0]].values[0]) + mdx
			dy =   (df[df.index <= pre_win[1]][cxyz[1]].values[-1] - \
					df[df.index >= pos_win[0]][cxyz[1]].values[0]) + mdy
			dz =   (df[df.index <= pre_win[1]][cxyz[2]].values[-1] - \
					df[df.index >= pos_win[0]][cxyz[2]].values[0]) + mdz

			# Update Dataframe entries 
			df.loc[pind,cxyz[0]] += dx
			df.loc[pind,cxyz[1]] += dy
			df.loc[pind,cxyz[2]] += dz

			mods.append([pre_win[1],pos_win[0],modx,mody,modz])


		df_mod = pd.DataFrame(mods,columns=['gstart','gstop','xmod','ymod','zmod'])

	return df, df_mod


### STEP3 Supporting Routines ###

def rotate(xy,Rmat,xy0=np.array([0.,0.])):
	"""
	Conduct a rotation on flattened (2,m) x,y array using a 
	pre-constructed Rotation matrix (Rmat)
	Allow option for an xy-origin (xy0) (2,) array
	"""
	# Ensure that xy is a numpy nd-array
	xy = np.asarray(xy)
	# Sanity check dimensions of input xy
	if xy.shape[0] != 2:
		if xy.shape[0] > xy.shape[1]:
			xy = xy.T
			trans=True
		else:
			warn('Invalid dimensions of xy: %d,%d'%(xy.shape))
	if Rmat.shape != (2,2):
		warn('Invalid dimensions of Rmat: %d,%d'%(Rmat.shape))
	# All else good, apply origin reference point if desired
	xy[0,:] -= xy0[0]
	xy[1,:] -= xy0[0]
	# Conduct rotation via matrix operation: Rmat*xy
	TS = np.matmul(Rmat,xy)
	# Return Transpose for list-format reading if the input xy was in the same format
	if trans:
		return TS.T
	# Otherwise return a (possibly large) collection 
	else:
		return TS

def rotaxy(theta,xy_cols,tol=1e-4):

	R11 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	R12 = np.round(-1.*np.sin(theta),int(-1*np.log10(tol)))
	R21 = np.round(np.sin(theta),int(-1*np.log10(tol)))
	R22 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	rmat = np.array([[R11,R12],[R21,R22]])
	
	TS = rotate(xy_cols,rmat)	

	return TS


# def rotaR3(yaw,pitch,roll,data,tol=1e-4,mode='measure',outmode='cols'):
def rotaR3(theta,data,tol=1e-4,mode='measure',outmode='cols'):
	"""
	Conduct rotation on R3 data with defined rotation angles about the
	input data bases

	:: INPUTS ::
	:type yaw: float (radians)
	:param yaw: Amount of anti-clockwise rotation about the initial Z-basis pole
	:type pitch: float (radians)
	:param pitch: Amount of anti-clockwise rotation about the initial Y-basis pole
	:type roll: float (radians)
	:param roll: Amount of anti-clockwise rotation about the initial X-basis pole
	:type data: numpy.ndarray (3 by n) [for mode = 'measure']
						   or (3 by 3 by n) [for mode = 'covmat']
						   or (n by 6) [for mode = 'covcols']
	:param data: Initial basis X,Y,Z data coordinates
	:type tol: float
	:param tol: machine precision level
	:type mode: string
	:param mode: 'measure' or 'covariance' - dictates the matrix multiplication scheme to apply
			'measure'   	X' = np.matmul(R,X)) 				(X' = RX)	Rotation of Data Matrix Mode
			'covmat'		C' = np.matmul(R,np.matmul(C,R.T))  (C' = RCRt) Rotation of Covariance Matrix Mode
					this mode assumes input is in a 3 by 3  by n format
			'covcols'		C' = np.matmul(R,np.matmul(C,R.T))  (C' = RCRt) Rotation of Covariance Matrix Mode
					this mode assumes input is in a 6 by n format where the columns are assigned as follows:
					
					0 = sXX;1 = sXY;2 = sXZ
					1		3 = sYY;4 = sYZ
					2		4		5 = sZZ;
					
					
					
	:type outmode: string
	:param outmode: Options are 'cols' or 'mats' - only applies for mode = 'covmat' OR 'covcols'
			'cols' return a flattened representation of the upper triangle of the output covariance matrix: C'
			'mats' return a full 3 by 3 by n numpy.ndarray set of covariance matrices where the 3rd axis is the sampling index
	"""

	# COMPOSE THE ROTATION MATRIX
	# for i_,theta in enumerate([yaw,pitch,roll]):
	# 	R11 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	# 	R12 = np.round(-1.*np.sin(theta),int(-1*np.log10(tol)))
	# 	R21 = np.round(np.sin(theta),int(-1*np.log10(tol)))
	# 	R22 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	# 	# Do Yaw (Z-axis CC rotation)
	# 	if i_ == 0 and abs(theta) > tol:
	# 		Rz = np.array([[R11,R12,0.],[R21,R22,0.],[0.,0.,1.]])
	# 	# Do Pitch (Y-axis CC rotation)
	# 	elif i_ == 1 and abs(theta) > tol:
	# 		Ry = np.array([[R11,0.,R21],[0.,1.,0.],[R12,0.,R22]])
	# 	# Do Roll (X-axis CC rotation)
	# 	elif i_ == 2 and abs(theta) > tol:
	# 		Rx = np.array([[0.,0.,1.],[R11,R12,0.],[R21,R22,0.]])
	# Compose R3 Rotation Matrix
	R11 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	R12 = np.round(-1.*np.sin(theta),int(-1*np.log10(tol)))
	R21 = np.round(np.sin(theta),int(-1*np.log10(tol)))
	R22 = np.round(np.cos(theta),int(-1*np.log10(tol)))
	rmat = np.array([[R11,R12,0.],\
		             [R21,R22,0.],\
		             [0. ,0. ,1.]])

	# rmat = np.matmul(Rz,Ry)
	# rmat = np.matmul(rmat,Rx)

	# APPLY THE ROTATION MATRIX IN THE SPECIFIED SCHEME
	# Apply X' = RX for data measures
	if mode == 'measure':
		dat = data.T
		out = np.matmul(rmat,dat)
		out = out
	# Apply C' = RCRt
	elif mode == 'covmat':
		# Create a holder for rotated covariance tensors
		out = np.zeros(data.shape)
		# Iterate across covariance matrix entries and conduct the RCR' operation
		for i_ in range((data.shape[2])):
			dat = data[:,:,i_]
			out[:,:,i_] = np.matmul(rmat,np.matmul(dat,rmat))
	# Column vector representation of the upper triangle of the covariance matrix
	# and then apply the C' = RCRt operation
	elif mode == 'covcols':
		# Create a holder for for rotated covariance tensors
		out = np.zeros((3,3,data.shape[0]))
		for i_ in range((data.shape[0])):
			idat = data[i_,:]
			# Compose row-specific covariance matrix
			C = np.array([[idat[0],idat[1],idat[2]],
						  [idat[1],idat[3],idat[4]],
						  [idat[2],idat[4],idat[5]]])
			out[:,:,i_] = np.matmul(rmat,np.matmul(C,rmat))

	# Handle output format option if 'cols' is specified
	if outmode == 'cols' and mode in ['covmat','covcols']:
		out = np.array([out[0,0,:],out[0,1,:],out[0,2,:],out[1,1,:],out[1,2,:],out[2,2,:]])

	# Dump the final result as output
	return out

def make_R3_mat(alpha,beta,gamma):
	"""
	Create 3D rotation matrix
	:: INPUTS ::
	:alpha: rotation angle, counter clockwise around the x axis
	:beta: rotation angle, counter clockwise around the y axis
	:gamma: rotation angle, counterclockwise around the z axis
	"""
	# Define common elements
	ca = np.cos(alpha)
	sa = np.sin(alpha)
	cb = np.cos(beta)
	sb = np.sin(beta)
	cg = np.cos(gamma)
	sg = np.sin(gamma)

	# Define Individual Rotation Matrices
	rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
	ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
	rz = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
	# Matrix multiplication to create compounded rotation matrix
	rmat = rz.dot(ry).dot(rx)
	return rmat

def rotate_columns(S_E,S_N,S_U,alpha,beta,gamma):
	"""
	Rotate from a E-N-U right-handed basis to a specified X-Y-Z basis with
	rotation angles alpha, beta, and gamma


	:: INPUTS ::
	:type S_E: pandas.Series
	:param S_E: East displacement data
	:type S_N: pandas.Series
	:param S_N: North displacement data
	:type S_U: pandas.Series
	:param S_U: Vertical displacement data
	:type alpha: [rad] float
	:param alpha: counterclockwise rotation angle to apply about the E-basis
	:type beta: [rad] float
	:param beta: counterclockwise rotation angle to apply about the N-basis
	:type gamma: [rad] float
	:param gamma: counterclockwise rotation angle to apply about the Z-basis


	:: OUTPUT ::
	:rtype stLQT: obspy.stream.Stream()
	:return stLQT: Stream containing traces for L,Q,T rotated data. Channel names are re-flagged as ??[LQT]

	"""
	# Form ZEN --> LQT basis rotation matrix
	rmat = make_R3_mat(alpha,beta,gamma)
	# rmat = np.array([[np.cos(phi), -np.sin(phi)*np.sin(beta), -np.sin(phi)*np.cos(beta)],
	# 				 [np.sin(phi),  np.cos(phi)*np.sin(beta),  np.cos(phi),np.cos(beta)],
	# 				 [0.         , -np.cos(beta),              np.sin(beta)            ]])
	# ENZdata = np.zeros((3,stream[0].stats.npts))
	Udata = S_U.values
	Edata = S_E.values
	Ndata = S_N.values
	# Form data column vector
	try:
		ENUdata = np.concatenate([Edata[:,np.newaxis],
								  Ndata[:,np.newaxis],
								  Udata[:,np.newaxis]],
								  axis=1)
	except ValueError:
		print('Something went wrong!')
		print('Udata size: %d'%(len(Zdata)))
		print('Edata size: %d'%(len(Edata)))
		print('Ndata size: %d'%(len(Ndata)))
	# Do transpose to align with rotation matrix dimensions
	if ENUdata.shape[0] != 3 and ENUdata.shape[1] == 3:
		ENUdata = ENUdata.T
	# Calculate rotated data
	XYZdata = rmat.dot(ENUdata)
	# Convert rotated data array back into a DataFrame for output
	df_out = pd.DataFrame({'mX':XYZdata[0,:],'mY':XYZdata[1,:],'mZ':XYZdata[2,:]},\
						   index=S_U.index)

	return df_out

def rotate_ENU_cov(df_in,alpha,beta,gamma,update=1000,savepoints=None,save_loc=None,cm={'see':'sde(m)','sne':'sdne(m)','sue':'sdeu(m)','snn':'sdn(m)','snu':'sdun(m)','suu':'sdu(m)'}):
	"""
	rotate East-North-Up covariance matrices stored tabular format with an arbitrary R3 rotation.
	This version of the code assumes that input data are in (co)variance**0.5 format
		this is inherited from the RTKLIB post-processing output conventions

	Note: This really is just a R3 matrix rotation for a dextral orthogonal basis using the form
			C(rotated) = R dot C dot R**-1
			with C the covariance matrix, R the rotation matrix, and R**-1 the inverted rotation matrix

	:: INPUTS ::
	:type df_in: pandas.DataFrame
	:param df_in: input DataFrame that contains 6 columns that define a symmetric R3 covariance matrix
	:param alpha: counter-clockwise rotation angle to apply about the East basis vector
	:param beta: counter-clockwise rotation angle to apply about the North basis vector
	:param gamma: counter-clockwise rotation angle to apply about the Up basis vector
	:param update: how often to print an update on rotation progress (this can take awhile) - every 'update' iterations
	:param savepoints: specify when to create saved temporary files as every 'savepoints' iterations
		(NTS 11. Jan 2022) BUG: keep this as None for now, with save-points the last segmet is not saved to output
	:param save_loc: path for where to write temporary save files
	:param cm: Covariance Matrix Mask Dictionary. Specify columns names in df_in that correspond to the following

			Covariance matrix C is given as:
			C = S'S (S-transpose matmul S)
			Composed of "standard error" matrix S
				| see  sne  sue |     s = sigma / sample uncertainty
			S = | sne  snn  snu |     e = east; n = north; u = up
				| sue  snu  suu | 

			Unique entries in S are used as keys for the mask (see default as example)

	:: OUTPUT ::
	:return df_out: DataFrame with the same index as df_in and uniquely labeled columns from C(rotated)
						 | vxx  vxy  vxz |
			C(rotated) = | vxy  vyy  vyz |   ===> df_out = |index vxx vxy vxz vyy vyz vzz|
						 | vxz  vyz  vzz |				  |IND    ##  ##  ##  ##  ##  ##|

	"""
	rmat = make_R3_mat(alpha,beta,gamma)
	# Create holder lists
	ind = []; vxx = []; vxy = []; vxz = []; vyy = []; vyz = []; vzz = []
	for i_ in range(len(df_in)):
		if i_%update == 0:
			print('...rotation %d/%d (%.1f%s) complete...'%(i_,len(df_in),(i_/len(df_in))*100,'%'))
		# Get entry
		dfi = df_in.iloc[i_]
		# Extract uncertainties and square to get (co)variance array
		CMAT = np.array([[dfi[cm['see']]**2,dfi[cm['sne']]**2,dfi[cm['sue']]**2],\
						 [dfi[cm['sne']]**2,dfi[cm['snn']]**2,dfi[cm['snu']]**2],\
						 [dfi[cm['sue']]**2,dfi[cm['snu']]**2,dfi[cm['suu']]**2]])
		# Do R.C.R**-1 matrix multiplication to rotate covariance matrix
		rCMAT = rmat.dot(CMAT).dot(np.linalg.inv(rmat))
		ind.append(dfi.name)
		vxx.append(rCMAT[0,0])
		vxy.append(rCMAT[0,1])
		vxz.append(rCMAT[0,2])
		vyy.append(rCMAT[1,1])
		vyz.append(rCMAT[1,2])
		vzz.append(rCMAT[2,2])
		if savepoints is not None and i_ > 0:
			if save_loc is None:
				save_loc = '.'
			if i_ % savepoints == 0:
				print('Savepoint Hit. Writing to %s'%(save_loc))
				df_tmp = pd.DataFrame({'vxx':vxx,'vxy':vxy,'vxz':vxz,'vyy':vyy,'vyz':vyz,'vzz':vzz},index=ind)
				df_tmp.to_csv(os.path.join(save_loc,'tmp_rotate_ENU_cov_DF_%d_%d.csv'%(df_tmp.index.values[0],i_)))				
				# Reset holders to clear memory
				ind = []; vxx = []; vxy = []; vxz = []; vyy = []; vyz = []; vzz = []
	if savepoints is not None:
		print('Processing complete! Recalling savepoint data & recompiling df_out')
		flist = glob(save_loc+'/tmp_rotate_ENU*.csv')
		for j_,f_ in enumerate(flist):
			if j_ == 0:
				df_out = pd.read_csv(f_,index_col=0,parse_dates=True)
			else:
				df_out = pd.concat([df_out,pd.read_csv(f_,index_col=0,parse_dates=True)],axis=0,ignore_index=False)

	else:
		df_out = pd.DataFrame({'vxx':vxx,'vxy':vxy,'vxz':vxz,'vyy':vyy,'vyz':vyz,'vzz':vzz},index=ind)
	return df_out


### STEP4 Supporting Functions ###



def df_centered_window(df,t0,wl):
	"""
	Conduct window-centered sampling of a data-frame with a DatetimeIndex
	and return a copied view of the DataFrame subset that can be independently
	manipulated

	:: INPUTS ::
	:param df: pandas.DataFrame with a DatetimeIndex type index
	:param t0: pandas.Timestamp specifing the window centered time
	:param wl: pandas.Timedelta specifing the length of the window

	:: OUTPUT ::
	:return idf: pandas.DataFrame sampled with t0 - 0.5wl <= df.index < t0 + 0.5wl
	"""
	ts = t0 - 0.5*wl
	te = t0 + 0.5*wl
	IDX = (df.index >= ts) & (df.index < te)
	idf = df.copy()[IDX]
	return idf

def detect_dt(index, method=np.median,already_sorted=True):
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

	:: OUTPUT ::
	:return dt_hat: value output by 'method'
	"""
	if not already_sorted:
		iidx = index.copy().sort_values()
	else:
		iidx = index.copy()
	dtv = (iidx[1:] - iidx[:-1]).total_seconds()
	dt_hat = method(dtv)
	return dt_hat


def df_rolling_wls_fit(df,wl=pd.Timedelta(2,unit='h'),Y='mX',vY='vxx',min_pct=0.75,dt_kwargs={'method':np.median,'already_sorted':True}):
	"""
	Conduct a centered rolling window WLS linear fitting to a DataFrame

	:: INPUTS ::
	:type df: pandas.DataFrame
	:param df: DataFrame with a DatetimeIndex and fields Y and vY
	:type wl: pandas.Timedelta
	:param wl: Sampling window length
	:type Y: str, column in df
	:param Y: dependent variable means to conduct fittings on
	:type vY: str, column in df
	:param vY: dependent variable variance to use in weighting
	:type min_pct: fract
	:param min_pct: minimum [fractional] percent of windowed data

	"""
	# Create Holder Dict
	df_out
	# Estimate n-samples for a full window
	dt_hat = detect_dt(df.index,**dt_kwargs)
	nptsF = int(wl/pd.Timedelta(dt_hat,unit='sec'))
	for i_ in tqdm(df.index):
		# First check if there's enough data in general
		if i_ - df.index.min() >= min_pct*wl and df.index.max() - i_ >= min_pct*wl:
			# Then sample
			idf = df_centered_window(df,i_,wl)
			# Now check for valid numer of points
			if np.sum(idf[Y].notna())/len(idf[Y]) > min_pct:
				dtv = (idf.index - i_).total_seconds()
				dyuv = idf[Y].values
				dyvv = idx[vY].values
				IDX = np.isfinite(dtv) & np.isfinite(dyuv) & np.isfinite(dyvv)
				mod,cov = np.polyfit(dtv[IDX],dyuv[IDX],1,w=dyvv**(-0.5),cov=True)


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
            # Conduct Velocity Modelling
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


