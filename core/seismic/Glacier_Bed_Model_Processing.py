import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from glob import glob
from scipy.io import loadmat

def list_fields(elab_dict):
	# Convenience method to list fields once an elaboration database is loaded into python
	print(list(elab_dict.keys()))

def vect_rotate(X,Y,THETA):
	rmat = np.array([[np.cos(THETA),-np.sin(THETA)],\
					 [np.sin(THETA), np.cos(THETA)]])
	Q = np.zeros(len(X))
	T = np.zeros(len(X))
	for i_ in range(len(X)):
		qt = np.matmul(rmat,np.array([X[i_],Y[i_]]))
		Q[i_] = qt[0]
		T[i_] = qt[1]

	return Q,T

def grid_rotate(XX,YY,THETA):
	"""
	Conduct an element-wise rotation of numpy.meshgrid() rendered grid coordinates
	:: INPUTS ::
	:type XX: numpy.ndarray (dim = 2)
	:param XX: X-axis coordinates
	:type YY: numpy.ndarray (dim = 2)
	:param YY: Y-axis coordinates
	:type THETA: float (rad)
	:param THETA: Counterclockwise rotation angle to apply

	:: OUTPUTS ::
	:rtype QQ: numpy.ndarray (dim = 2)
	:return QQ: New "X-axis" coordinate grid
	:rtype TT: numpy.ndarray (dim = 2)
	:return TT: New "Y-axis" coordinate grid
	"""
	rmat = np.array([[np.cos(THETA),-np.sin(THETA)],\
					 [np.sin(THETA), np.cos(THETA)]])

	QQ = np.zeros(XX.shape)
	TT = np.zeros(XX.shape)
	for i_ in range(XX.shape[0]):
		for j_ in range(XX.shape[1]):
			qt = np.matmul(rmat,np.array([XX[i_,j_],YY[i_,j_]]))
			QQ[i_,j_] = qt[0]
			TT[i_,j_] = qt[1]

	return QQ,TT

def parse_OHVSR_proj_mfile(mfile,inst=None):
	"""
	Read a *.m file defining the start of an OpenHVSR Processing Toolbox Project
	Return a DataFrame containing station locations, processing time bounds, and instrument type, if supplied

	:: INPUTS ::
	:type mfile: str
	:param mfile: OpenHVSR project *.m file to parse
	:type inst: None or str or list
	:param inst: name of instrument type common to the input mfile, if a list, must match entries

	:: OUTPUT ::
	:rtype df_meta: pandas.DataFrame
	:return df_meta: DataFrame containing extracted metadata from the project.m file
	"""
	f_ = open(mfile)
	L_ = f_.readlines()
	idx = []; mE = []; mN = []; mZ = []; sta = []; ts = []; te = []; typ = []
	if inst is not None:
		if isinstance(inst,list) and len(inst) == len(L_):
			typ = inst
	for i_,l_ in enumerate(L_):
		if 'SURVEYS' in l_:
			SL1_ = l_.split(';')
			idx.append(int(SL1_[0].split(',')[0].split('{')[-1]))
			mE.append(int(SL1_[0].split(',')[1].split('[')[-1]))
			mN.append(int(SL1_[0].split(',')[2]))
			mZ.append(int(SL1_[0].split(',')[3].split(']')[0]))
			sta.append(SL1_[1].split('_')[1])
			ts.append(pd.Timestamp(float(SL1_[1].split('_')[2]),unit='s'))
			te.append(pd.Timestamp(float(SL1_[1].split('_')[3][:-5]),unit='s'))
			if not isinstance(inst,list):
				typ.append(inst)
	if inst is not None:
		df_meta = pd.DataFrame({'DAS':sta,'type':typ,'start_time':ts,'end_time':te,'mE':mE,'mN':mN,'mZ':mZ})
	else:
		df_meta = pd.DataFrame({'DAS':sta,'start_time':ts,'end_time':te,'mE':mE,'mN':mN,'mZ':mZ})
	df_meta.index = idx
	f_.close()
	return df_meta

def parse_matfile(elab_db_mat):
	"""
	Extract the HVSR curves (master & ensemble), frequencies, and peaks

	:: INPUT :: 
	:type elab_db_mat: str
	:param elab_db_mat: *.mat file for a specific OpenHVSR elaboration 

	:: OUTPUTS :: 
	:rtype user_f0: float OR None
	:return user_f0: IF PRESENT, user-defined fundamental frequency (f0) HV pick
	:rtype auto_f0: float
	:return auto_f0: Automatically picked global HV maximum
	:rtype added_f0: float, list, OR None
	:return added_f0: Additional f0 picks present in the elaboration
	:rtype df_HV: pandas.DataFrame
	:return df_HV: DataFrame containing raw HV curves (horizontally averaged, as specified in the elaboration)
				   with the Fourier Frequencies as the index and the curve integer indices as the columns
				   I.e., a set of HV(f) column vectors.
	
	:: TODO ::
	Extract EV and NV to allow later mixing. Should occur around the point #^v^v
	
	"""
	# Load *.mat file
	data = loadmat(elab_db_mat)

	# Get Picked & Automatic Peaks, Assign As 'None'
	user_f0 = float(data['iDTB__hvsr__user_main_peak_frequence'])
	if np.isnan(user_f0):
		user_f0 = None
	auto_f0 = float(data['iDTB__hvsr__auto_main_peak_frequence'])
	if np.isnan(auto_f0):
		auto_f0 = None
	added_f0 = data['iDTB__hvsr__user_additional_peaks']

	# Extract & Package the Reference HVSR Curve
	# Use of the non-"_full" fields includes the filtering done in OpenHVSR
	HV_Ref_mean = data['iDTB__hvsr__curve']
	HV_Ref_std = data['iDTB__hvsr__standard_deviation']
	HV_Ref_p95 = data['iDTB__hvsr__confidence95']
	df_HV_Ref = pd.DataFrame({'mean':HV_Ref_mean[:,0],'std':HV_Ref_std[:,0],'p95':HV_Ref_p95[:,0]})

	# Extract Raw HVSR Curves
	#^v^v
	HV_RAW = data['iDTB__hvsr__HV_all_windows']
	# EV_RAW = data['iDTB__hvsr__EV_all_windows']
	tmp_dict = {}
	for i_ in range(HV_RAW.shape[1]):
		tmp_dict.update({(i_):HV_RAW[:,i_]})
	df_HV_raw = pd.DataFrame(tmp_dict)

	# Reconstitute the Frequency Vector
	nfreq = len(HV_Ref_mean)
	fmin = float(data['iDTB__section__Min_Freq'])
	fmax = float(data['iDTB__section__Max_Freq'])
	ffreqs = np.linspace(fmin,fmax,nfreq)

	# Package Data into a DataFrame for export
	df_HV = pd.concat([df_HV_Ref,df_HV_raw],axis=1)
	df_HV.index = ffreqs
	df_HV.index.name = 'freq'

	return user_f0, auto_f0, added_f0, df_HV

def plot_hv_result(elab_db_mat,dTHETA=None):
	data = loadmat(elab_db_mat)
	# Get Picked & Automatic Peaks, Assign As 'None'
	user_f0 = float(data['iDTB__hvsr__user_main_peak_frequence'])
	if np.isnan(user_f0):
		user_f0 = None
	auto_f0 = float(data['iDTB__hvsr__auto_main_peak_frequence'])
	if np.isnan(auto_f0):
		auto_f0 = None
	added_f0 = data['iDTB__hvsr__user_additional_peaks']

	# Extract & Package the Reference HVSR Curve
	# Use of the non-"_full" fields includes the filtering done in OpenHVSR
	HV_Ref_mean = data['iDTB__hvsr__curve']
	HV_Ref_std = data['iDTB__hvsr__standard_deviation']
	HV_Ref_p95 = data['iDTB__hvsr__confidence95']
	df_HV_Ref = pd.DataFrame({'mean':HV_Ref_mean[:,0],'std':HV_Ref_std[:,0],'p95':HV_Ref_p95[:,0]})

	# Extract Raw HVSR Curves
	#^v^v
	HV_RAW = data['iDTB__hvsr__HV_all_windows']
	# EV_RAW = data['iDTB__hvsr__EV_all_windows']
	tmp_dict = {}
	for i_ in range(HV_RAW.shape[1]):
		tmp_dict.update({(i_):HV_RAW[:,i_]})
	df_HV_raw = pd.DataFrame(tmp_dict)

	# Reconstitute the Frequency Vector
	nfreq = len(HV_Ref_mean)
	fmin = float(data['iDTB__section__Min_Freq'])
	fmax = float(data['iDTB__section__Max_Freq'])
	ffreqs = np.linspace(fmin,fmax,nfreq)

	# Package Data into a DataFrame for export
	df_HV = pd.concat([df_HV_Ref,df_HV_raw],axis=1)
	df_HV.index = ffreqs
	df_HV.index.name = 'freq'

	plt.figure()
	plt.semilogx(ffreqs,df_HV_raw,color='gray',alpha=0.25)
	plt.semilogx(ffreqs,df_HV_Ref['mean'],'k')
	plt.fill_between(ffreqs,df_HV_Ref['mean']-df_HV_Ref['p95'],\
							df_HV_Ref['mean']+df_HV_Ref['p95'],\
							color='b',alpha=0.5)
	if user_f0 is not None:
		plt.plot(np.ones(2)*user_f0,[0,10],'r-',label='Manual Pick')
	if auto_f0 is not None:
		plt.plot(np.ones(2)*auto_f0,[0,10],'b-',label='Auto Pick')
	if added_f0 is not None:
		for i_ in added_f0:
			if i_ == 0:
				ilbl = 'Additional Picks'
			else:
				ilbl = None
			plt.plot(np.ones(2)*i_,[0,10],'c-',label=ilbl)

	
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('HV Ratio')
	plt.legend()

def fetch_f0_a0_from_raw(f0_pick,df_HV,bounds=0.25,smoothing=None):
	"""
	Conduct "supervised" mean & uncertainty estimation of the fundamental frequency (f0) with
	the ansatz of a normal distribution. f0 values are estimated for each raw HV curve to create 
	an estimate set and statistics are conducted from there.

	f0 estimates are picked as the global maximum for an individual HV curve within f0_pick +/-bounds

	Options to smooth each HV curve via an integer-index-lengthed window using the pandas.DataFrame.rolling() method

	:: INPUTS ::
	:type f0_pick: float
	:param f0_pick: seed pick for f0
	:type df_HV: pandas.DataFrame
	:param df_HV: DataFrame containing HV curves, index as Fourier frequencies, columns as curve indices
	:type bounds: float
	:param bounds: Maximum +/- boundaries from f0_pick to include from curves
	:type smoothing: None or int
	:param smoothing: window length for boxcar smoothing. Smoothing is window centered.

	:: TODO ::
	--> Options to include other statistical frameworks
	--> Options for different smoothing windows

	"""
	# Get number of raw HVSR curves
	n_raw = df_HV.shape[1] - 3
	f0_ests = []; a0_ests = []
	DF = df_HV.index[1] - df_HV.index[0]
	for i_ in range(n_raw):
		iS = df_HV[i_][(df_HV.index >= f0_pick - bounds) & (df_HV.index <= f0_pick + bounds)]
		# Option to apply smoothing 
		if smoothing is not None:
			# Currently, just offer a rolling mean 
			iSS = iS.rolling(int(smoothing)).mean()
			iSS.index = iS.index - DF*(int(smoothing)/2)
		# Otherwise alias iS to iSS
		else:
			iSS = iS
		f0_ests.append(iSS.idxmax())
		a0_ests.append(iSS.max())
	return f0_ests, a0_ests

def fit_gaussian(f0_ests):
	"""
	Convenience method around the scipy.stats.normal.fit() method
	:: INPUT ::
	:type f0_ests: list of float
	:param f0_ests: list of individual f0 estimates

	:: OUTPUTS ::
	:rtype mu: float
	:return mu: mean f0 estimate from the input set
	:rtype sigma: float
	:return sigma: standard deviation for input f0 set
	"""
	mu,sigma = spst.norm.fit(f0_ests)
	return mu, sigma

def plot_model_fit(df_HV,f0_ests,a0_ests,f0_pick,bounds,mu,sigma):
	"""
	Module-tailored plotting routine for visualizing statistics
	Uses semilogx() plotting & generates a new figure.

	Does not include a matplotlib.pyplot.show() call at the end to allow further
	command interaction in lieu of a %pylab instance in ipython or %pylab notebook in jupyter
	"""
	plt.figure()
	for i_ in range(df_HV.shape[1] - 3):
		plt.semilogx(df_HV[i_],alpha=0.2)
	plt.semilogx(df_HV['mean'],'k')
	plt.semilogx(df_HV['mean'] + df_HV['p95'],'k--')
	plt.semilogx(df_HV['mean'] - df_HV['p95'],'k--')
	plt.semilogx(np.ones(2)*f0_pick,[0,np.max(a0_ests)*1.2],'r-',label='manual')
	plt.semilogx(np.ones(2)*mu,[0,np.max(a0_ests)*1.2],'b-',label='mean')
	plt.semilogx(np.ones(2)*(mu - sigma),[0,np.max(a0_ests)*1.2],'b:',label='std')
	plt.semilogx(np.ones(2)*(mu + sigma),[0,np.max(a0_ests)*1.2],'b:')
	plt.legend()

def process_enumeration(enum_dir,proj_mfile,inst=None,bounds=1,smoothing=10, pick_pref='manual'):
	"""
	Process a full set of enumerations associated with an input project file for a given
	OpenHVSR Processing Toolbox project

	:: INPUTS ::
	:type enum_dir: str
	:param enum_dir: system-appropriate formatted path to the enumeration associated with the target project
	:type proj_mfile: str
	:param proj_mfile: path/file for the appropriate OpenHVSR project *.m file
	:type pref_pick: str
	:param pref_pick: 'manual' or 'automatic' which type of pick to give preference to for an individual HV realization

	# Subroutine Parameters
	:inst: See parse_OHVSR_proj_mfile()
	:bounds: See fetch_f0_a0_from_raw()
	:smoothing: See fetch_f0_a0_from_raw()

	:: OUTPUT ::
	:rtype df_out: pandas.DataFrame
	:return df_out: DataFrame containing manual and gaussian statistical estimates of f0 for each station/realization in 
					the input enumeration / project. Metadata on the preferred pick (pref_pick) type and instrument type



	"""
	df_meta = parse_OHVSR_proj_mfile(proj_mfile,inst=inst)
	flist = glob(os.path.join(enum_dir,'Elaboration_database_*.mat'))
	idx = []; f0_gau_u = []; f0_gau_o = []; f0_man = []; ptype = []; elab = []; proj = []
	for f_ in flist:
		elab.append(f_)
		proj.append(proj_mfile)
		idx.append(int(f_.split('of')[0].split('_')[-1]))
		print('processing {}'.format(df_meta[df_meta.index==idx[-1]]))
		uf0,af0,aaf0,df_HV = parse_matfile(f_)
		if pick_pref == 'manual' and uf0 is not None:
			pick = uf0
			ptype.append('manual')
		else:
			pick = af0
			ptype.append('automatic')
		f0_ests, a0_ests = fetch_f0_a0_from_raw(pick,df_HV,bounds=bounds,smoothing=smoothing)
		u_f0, o_f0 = fit_gaussian(f0_ests)
		f0_gau_u.append(u_f0)
		f0_gau_o.append(o_f0)
		f0_man.append(pick)
	df_f0 = pd.DataFrame({'f0_gau_u':f0_gau_u,'f0_gau_o':f0_gau_o,'f0_man':f0_man,'f0_man_type':ptype,\
						  'elaboration':elab,'project':proj})
	df_f0.index = idx

	df_out = pd.concat([df_meta,df_f0],axis=1,ignore_index=False)
	return df_out

def quarter_wavelength_processing(enum_dir,proj_mfile,Vs_u,Vs_o,inst=None,bounds=1,smoothing=10,pick_pref='manual'):
	"""
	Conduct processing on the raw curves & manual/automatic picks to extract gaussian statistics on peak frequency samples
	using the Quarter Wavelength empirical relationship for a 2 layer system with a compliant sheet overlying a rigid half-space
	E.g., Bard & Bouchon (1980a,b)

	### Roughly in LaTeX markup syntax

	$ \\mu_H = {V_S}over{4 f_0}$
	$ \\sigma_H = sqrt{\\mu_H^2((V_{S,\\sigma}/V_{S,\\mu})^2 + (f_{0,\\sigma}/f_{0,\\mu})^2)}

	:: INPUTS ::
	:type enum_dir: str
	:param enum_dir: path for target OpenHVSR enumeration file directory - passed to process_enumeration() module subroutine
	:type proj_mfile: str
	:param proj_mfile: path/file for target OpenHVSR project *.m file - passed to process_enumeration()
	:type Vs_u: float
	:param Vs_u: mean shear velocity for the compliant layer
	:type Vs_o: float
	:param Vs_o: standard deviation for shear velocity in the compliant layer

	# Subroutine Parameters #
	:inst: instrument type - See process_enumeration()
	:bounds: f0 search boundaries - See process_enumeration()
	:smoothing: See process_enumeration()
	:pick_pref: See process_enumeration()

	:: OUTPUT ::
	:rtype df_OUT: pandas.DataFrame
	:return df_OUT: DataFrame containing compiled f0 and compliant layer thickness estimates, using appropriate error
					propagation rules 
	"""
	df_out = process_enumeration(enum_dir,proj_mfile,inst=inst,bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
	f0gu = df_out['f0_gau_u'].values
	f0go = df_out['f0_gau_o'].values
	f0m = df_out['f0_man'].values

	hgu = Vs_u/(4.*f0gu)
	hgo = (hgu**2*((Vs_o/Vs_u)**2 + (f0go/f0gu)**2))**0.5

	hmu = Vs_u/(4.*f0m)

	df_H = pd.DataFrame({'H_gau_u':hgu,'H_gau_o':hgo,'H_man':hmu})
	df_H.index = df_out.index
	df_OUT = pd.concat([df_out,df_H],axis=1,ignore_index=False)
	return df_OUT

def run_pick_statistics(Vs_u = 3451/1.93, Vs_o = 168, bounds = 1, smoothing = 10, pick_pref='manual'):
	"""
	A pre-defined processing wrapper for the SGGS projects on hand
	"""
	ROOT_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY/OpenHVSR_Project'
	proj_files = ['RD_Data/OpenHVSR_SS1_proj_100Hz.m',\
				  'RD_Data/OpenHVSR_SGGS_SS_proj_100Hz.m',\
				  'RD_Data/OpenHVSR_SGGS_MF_proj_100Hz.m',\
				  'RD_Data/OpenHVSR_Reflect_proj_100Hz.m',\
				  'RD_Data/OpenHVSR_Refract_proj_100Hz.m']
	enum_dirs =  ['elaborations/SaskSeis_Array',\
				  'elaborations/SGGS_SmartSolo',\
				  'elaborations/SGGS_MagseisFairfield',\
				  'elaborations/SGGS_Reflection_Line',\
				  'elaborations/SGGS_Refraction_Line']
	insts = ['DC17','SS19','MF19','DC19','RL19']
	for I_ in range(5):
		print("!!Processing {} - {}".format(enum_dirs[I_],proj_files[I_]))
		df_IOUT = quarter_wavelength_processing(os.path.join(ROOT_DIR,enum_dirs[I_]),\
												os.path.join(ROOT_DIR,proj_files[I_]),\
												Vs_u,Vs_o,inst=insts[I_],\
												bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
		if I_ == 0:
			df_MAIN = df_IOUT.copy()
		else:
			df_MAIN = pd.concat([df_MAIN,df_IOUT],axis=0,ignore_index=True)
	return df_MAIN

def create_meshes(df_M,mesh_method='linear',dX=10,THETA=-np.pi/4,ODG={'mE':487785.,'mN':5777343.},ODT={'AAp':5970,'BBp':200}):
	"""

	"""
	# Iterate across Thickness Estimate Parameters
	Z_gau_u_min = []
	Z_gau_u_mea = []
	Z_gau_u_max = []
	H_gau_o_min = []
	H_gau_o_mea = []
	H_gau_o_max = []
	X = []; Y = []; Z = []; S = []; E = []; N = []
	for i_ in df_M['DAS'].unique():
		IDX = df_M['DAS']==i_
		S.append(i_)
		E.append(df_M[IDX]['mE'].mean())
		N.append(df_M[IDX]['mN'].mean())
		X.append(df_M[IDX]['mE'].mean() - ODG['mE'])
		Y.append(df_M[IDX]['mN'].mean() - ODG['mN'])
		Z.append(df_M[IDX]['mZ'].mean())
		Z_gau_u_min.append((df_M[IDX]['mZ'] - df_M[IDX]['H_gau_u']).min())
		Z_gau_u_mea.append((df_M[IDX]['mZ'] - df_M[IDX]['H_gau_u']).mean())
		Z_gau_u_max.append((df_M[IDX]['mZ'] - df_M[IDX]['H_gau_u']).max())
		H_gau_o_min.append(df_M[IDX]['H_gau_o'].min())
		H_gau_o_mea.append(df_M[IDX]['H_gau_o'].mean())
		H_gau_o_max.append(df_M[IDX]['H_gau_o'].max())
	# Rotate Values
	A,B = vect_rotate(X,Y,THETA)
	# And reference to transects
	A += ODT['AAp']
	B += ODT['BBp']
	AA,BB = np.meshgrid(np.arange(A.min(),A.max(),dX),np.arange(B.min(),B.max(),dX))
	# Convert back into intial reference frame
	# Undo Rotation of Grid
	XX,YY = grid_rotate(AA - ODT['AAp'],BB - ODT['BBp'],-1*THETA)
	EE = XX + ODG['mE']
	NN = YY + ODG['mN']
	# Ice Surface
	CC = griddata((A,B),Z,(AA,BB),method=mesh_method)
	# Bed Models
	ZZ_gu_min = griddata((A,B),np.array(Z_gau_u_min,dtype=float),(AA,BB),method=mesh_method)
	ZZ_gu_mea = griddata((A,B),np.array(Z_gau_u_mea,dtype=float),(AA,BB),method=mesh_method)
	ZZ_gu_max = griddata((A,B),np.array(Z_gau_u_max,dtype=float),(AA,BB),method=mesh_method)
	# Error Models
	SS_go_min = griddata((A,B),np.array(H_gau_o_min,dtype=float),(AA,BB),method=mesh_method)
	SS_go_mea = griddata((A,B),np.array(H_gau_o_mea,dtype=float),(AA,BB),method=mesh_method)
	SS_go_max = griddata((A,B),np.array(H_gau_o_max,dtype=float),(AA,BB),method=mesh_method)

	points = {'sta':S,'A':A,'B':B,'E':E,'N':N,'Z':Z,\
			  'Zmin':Z_gau_u_min,'Zmea':Z_gau_u_mea,'Zmax':Z_gau_u_max,\
			  'Smin':H_gau_o_min,'Smea':H_gau_o_mea,'Smax':H_gau_o_max}
	points = pd.DataFrame(points)
	surfs = {'AA':AA,'BB':BB,'EE':EE,'NN':NN,'ZZ':CC,\
			 'ZZmin':ZZ_gu_min,'ZZmea':ZZ_gu_mea,'ZZmax':ZZ_gu_max,\
			 'SSmin':SS_go_min,'SSmea':SS_go_mea,'SSmax':SS_go_max}

	return points,surfs

def model_hydropotential(surfs,surf_key,bed_key,pice=916.7,ph2o=1000,gg=9.81):
	HI = surfs[surf_key] - surfs[bed_key]
	HP = pice*gg*HI + ph2o*gg*surfs[bed_key]# - surfs[bed_key].min())
	return HP




def plot_bed_model(points,surfs,target_layer='ZZmea',layer_label='Mean Bed Elevation (m)'):
	XX = surfs['AA']
	YY = surfs['BB']
	ZZ = surfs[target_layer]
	Contours = np.arange(np.nanmin(ZZ) - np.nanmin(ZZ)%5,np.nanmax(ZZ),5)
	plt.figure()
	plt.pcolor(XX,YY,ZZ)
	cbar = plt.colorbar()
	cbar.set_label(layer_label,rotation=270)
	plt.contour(XX,YY,ZZ,Contours,colors='k')
	plt.plot(points['A'],points['B'],'v',color='firebrick')
	plt.xlabel("Distance Along A-A' (m)")
	plt.ylabel("Distance Along C-C' (m)")


def write_ascii_raster(file,EE,NN,ZZ,nodata_value=-9999):
	# Open File, appending extension if needed
	if file.split('.')[-1] != '.ascii':
		fobj = open(file+'.ascii',"w")
	else:
		fobj = open(file,"w")
	# Write Header
	fobj.write('ncols %d\n'%(EE.shape[1]))
	fobj.write('nrows %d\n'%(EE.shape[0]))
	fobj.write('xllcorner %.3f\n'%(np.nanmin(EE)))
	fobj.write('yllcorner %.3f\n'%(np.nanmin(NN)))
	fobj.write('cellsize %.3f\n'%(np.nanmean(EE[:,1:] - EE[:,:-1])))
	fobj.write('nodata_value %.3f\n'%(nodata_value))

	# Iterate across input griddata
	for i_ in np.arange(EE.shape[0],0,-1)-1:
		for j_ in range(EE.shape[1]):
			z_ij = ZZ[i_,j_]
			# Check if nodata (handles NaN & Inf)
			if not np.isfinite(z_ij):
				z_ij = nodata_value
			# Write entry ij
			fobj.write('%.3f'%(z_ij))
			# If not done with the line, add a space
			if j_+1 < EE.shape[1]:
				fobj.write(' ')
			# If done with the line, but more lines are present, add a return
			elif i_+1 < EE.shape[0]:
				fobj.write('\n')
	# Finish Writing
	fobj.close()





def save_surfaces(surfs,save_dir='.',prefix='',**kwargs):
	for k_ in surfs.keys():
		out_file = os.path.join(save_dir,prefix+k_+'.npy')
		print('Saving '+out_file)
		np.save(out_file,surfs[k_],**kwargs)

