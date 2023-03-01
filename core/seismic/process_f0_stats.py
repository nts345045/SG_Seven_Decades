import os
from glob import glob
from scipy.io import loadmat
import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd

ROOT = os.path.join('..','..','..','..')
WFDIR = os.path.join(ROOT,'data','MINDSatUW','SEISMIC','Passive','ASCII')
PJDIR = os.path.join(ROOT,'data','MINDSatUW','SEISMIC','Passive','OpenHVSR')


def list_fields(elab_dict):
	# Convenience method to list fields once an elaboration database is loaded into python
	print(list(elab_dict.keys()))

def parse_OHVSR_proj_mfile(mfile):
	f_ = open(mfile)
	L_ = f_.readlines()
	idx = []; mE = []; mN = []; mZ = []; sta = []; ts = []; te = []
	for l_ in L_:
		if 'SURVEYS' in l_:
			SL1_ = l_.split(';')
			idx.append(int(SL1_[0].split(',')[0].split('{')[-1]))
			mE.append(int(SL1_[0].split(',')[1].split('[')[-1]))
			mN.append(int(SL1_[0].split(',')[2]))
			mZ.append(int(SL1_[0].split(',')[3].split(']')[0]))
			sta.append(SL1_[1].split('_')[1])
			ts.append(pd.Timestamp(float(SL1_[1].split('_')[2]),unit='s'))
			te.append(pd.Timestamp(float(SL1_[1].split('_')[3][:-5]),unit='s'))
	df_meta = pd.DataFrame({'DAS':sta,'start_time':ts,'end_time':te,'mE':mE,'mN':mN,'mZ':mZ})
	df_meta.index = idx
	f_.close()
	return df_meta


def parse_matfile(elab_db_mat):
	"""
	Extract the HVSR curves (master & ensemble), frequencies, and peaks
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
	HV_RAW = data['iDTB__hvsr__HV_all_windows']
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

def fetch_f0_a0_from_raw(f0_pick,df_HV,bounds=0.25,smoothing=None):
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
	mu,sigma = sp.stats.norm.fit(f0_ests)
	return mu, sigma


def plot_model_fit(df_HV,f0_ests,a0_ests,f0_pick,bounds,mu,sigma):
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


def process_enumeration(enum_dir,proj_mfile,bounds=1,smoothing=10, pick_pref='manual'):
	df_meta = parse_OHVSR_proj_mfile(proj_mfile)
	flist = glob(os.path.join(enum_dir,'Elaboration_database_*.mat'))
	idx = []; f0_gau_u = []; f0_gau_o = []; 
	f0_med = []; f0_p05 = []; f0_p95 = [];
	f0_man = []; ptype = []
	for f_ in flist:
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
		m_f0 = np.median(f0_ests)
		p05_f0 = np.quantile(f0_ests,0.05)
		p95_f0 = np.quantile(f0_ests,0.95)
		f0_gau_u.append(u_f0)
		f0_gau_o.append(o_f0)
		f0_med.append(m_f0)
		f0_p95.append(p95_f0)
		f0_p05.append(p05_f0)
		f0_man.append(pick)
	df_f0 = pd.DataFrame({'f0_gau_u':f0_gau_u,'f0_gau_o':f0_gau_o,'f0_med':f0_med,'f0_p05':f0_p05,'f0_p95':f0_p95,'f0_man':f0_man,'f0_man_type':ptype})
	df_f0.index = idx

	df_out = pd.concat([df_meta,df_f0],axis=1,ignore_index=False)
	return df_out


def quarter_wavelength_processing(enum_dir,proj_mfile,Vs_u,Vs_o,bounds=1,smoothing=10,pick_pref='manual'):
	df_out = process_enumeration(enum_dir,proj_mfile,bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
	f0gu = df_out['f0_gau_u'].values
	f0go = df_out['f0_gau_o'].values
	f0m = df_out['f0_man'].values
	f0med = df_out['f0_med'].values
	f0p05 = df_out['f0_p05'].values
	f0p95 = df_out['f0_p95'].values
	hgu = Vs_u/(4.*f0gu)
	hgo = (hgu**2*((Vs_o/Vs_u)**2 + (f0go/f0gu)**2))**0.5
	hmed = Vs_u/(4.*f0med)
	hp95 = (Vs_u + 1.96*Vs_o)/(4.*f0p05)
	hp05 = (Vs_u - 1.96*Vs_o)/(4.*f0p95)

	hmu = Vs_u/(4.*f0m)

	df_H = pd.DataFrame({'H_gau_u':hgu,'H_gau_o':hgo,'H_med':hmed,'H_p05':hp05,'H_p95':hp95,'H_man':hmu})
	df_H.index = df_out.index
	df_OUT = pd.concat([df_out,df_H],axis=1,ignore_index=False)
	return df_OUT


def run(Vs_u = 3451/1.95, Vs_o = 168, bounds = 1, smoothing = 10, pick_pref='manual'):
	# # ROOT_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY/OpenHVSR_Project'
	# proj_files = ['RD_Data/OpenHVSR_SS1_proj_100Hz.m',\
	# 			  'RD_Data/OpenHVSR_SGGS_SS_proj_100Hz.m',\
	# 			  'RD_Data/OpenHVSR_SGGS_MF_proj_100Hz.m',\
	# 			  'RD_Data/OpenHVSR_Reflect_proj_100Hz.m',\
	# 			  'RD_Data/OpenHVSR_Refract_proj_100Hz.m']
	# enum_dirs =  ['elaborations/SaskSeis_Array',\
	# 			  'elaborations/SGGS_SmartSolo',\
	# 			  'elaborations/SGGS_MagseisFairfield',\
	# 			  'elaborations/SGGS_Reflection_Line',\
	# 			  'elaborations/SGGS_Refraction_Line']
	# proj_file = os.path.join(PJDIR,'OpenHVSR_proj.m')
	# enum_dir = os.path.join(PJDIR,'Elaboration')
	# df_MAIN = quarter_wavelength_processing(enum_dir,proj_file,Vs_u,Vs_o,bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
	# # for I_ in range(5):
	# # 	print("!!Processing {} - {}".format(enum_dirs[I_],proj_files[I_]))
	# # 	df_IOUT = quarter_wavelength_processing(os.path.join(ROOT_DIR,enum_dirs[I_]),\
	# # 											os.path.join(ROOT_DIR,proj_files[I_]),\
	# # 											Vs_u,Vs_o,\
	# # 											bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
	# # 	if I_ == 0:
	# # 		df_MAIN = df_IOUT.copy()
	# # 	else:
	# # 		df_MAIN = pd.concat([df_MAIN,df_IOUT],axis=0,ignore_index=True)
	# return df_MAIN
	# ROOT_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY/OpenHVSR_Project'
	proj_files = ['OpenHVSR_SS1_proj_100Hz.m',\
				  'OpenHVSR_proj.m']
	enum_dirs =  ['SaskSeis_Array','Elaboration']
	# for I_ in range(2):
	# proj_file = os.path.join(PJDIR,'OpenHVSR_proj.m')
	# enum_dir = os.path.join(PJDIR,'Elaboration')
	# df_MAIN = quarter_wavelength_processing(enum_dir,proj_file,Vs_u,Vs_o,bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
	for I_ in range(2):
		print("!!Processing {} - {}".format(enum_dirs[I_],proj_files[I_]))
		df_IOUT = quarter_wavelength_processing(os.path.join(PJDIR,enum_dirs[I_]),\
												os.path.join(PJDIR,proj_files[I_]),\
												Vs_u,Vs_o,\
												bounds=bounds,smoothing=smoothing,pick_pref=pick_pref)
		if I_ == 0:
			df_MAIN = df_IOUT.copy()
		else:
			df_MAIN = pd.concat([df_MAIN,df_IOUT],axis=0,ignore_index=True)
	return df_MAIN

#### RUN PROCESSING ####
uVs = 3451.0/1.95
oVs = 168
ODIR = os.path.join(ROOT,'processed_data','seismic')
_df_ = run()
_df_.to_csv(os.path.join(ODIR,'HV_analysis_outs.csv'),header=True,index=False)