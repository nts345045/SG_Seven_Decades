import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

"""
Create example HVSR curves to add to Seven_Decade_Fig2 layout in the QGIS_Project
Create 
"""

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
	EV_RAW = data['iDTB__hvsr__EV_all_windows']
	NV_RAW = data['iDTB__hvsr__NV_all_windows']
	# HV Ratio Windows
	tmp_dict = {}
	for i_ in range(HV_RAW.shape[1]):
		tmp_dict.update({(i_):HV_RAW[:,i_]})
	df_HV = pd.DataFrame(tmp_dict)
	# NV Ratio Windows
	tmp_dict = {}
	for i_ in range(EV_RAW.shape[1]):
		tmp_dict.update({(i_):EV_RAW[:,i_]})
	df_EV = pd.DataFrame(tmp_dict)
	# NV Ratio Windows
	tmp_dict = {}
	for i_ in range(NV_RAW.shape[1]):
		tmp_dict.update({(i_):NV_RAW[:,i_]})
	df_NV = pd.DataFrame(tmp_dict)

	# Reconstitute the Frequency Vector
	nfreq = len(HV_Ref_mean)
	fmin = float(data['iDTB__section__Min_Freq'])
	fmax = float(data['iDTB__section__Max_Freq'])
	ffreqs = np.linspace(fmin,fmax,nfreq)

	# Package Data into a DataFrame for export
	df_HV_Ref.index = ffreqs
	df_HV_Ref.index.name = 'freq'
	# Propagate frequency index to 
	df_HV.index = df_HV_Ref.index
	df_EV.index = df_HV_Ref.index
	df_NV.index = df_HV_Ref.index

	OUT = {'user_f0':user_f0,'auto_f0':auto_f0,'added_f0':added_f0,\
		   'HV_ref':df_HV_Ref,'HV_raw':df_HV,'EV_raw':df_EV,'NV_raw':df_NV}

	return OUT

ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/VALLEY_GEOMETRY/OpenHVSR_Project'
OUT_DIR = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FIGURES/v11_render/supplement/FigureS4'

INDEX_FILE = ROOT+'/Outputs/Master_Estimate_1788pm168mps_b1_v3.csv'
# Load Master Index CSV
df_M = pd.read_csv(INDEX_FILE,parse_dates=True)

# exsta = np.array(['R2108','1110'])
exsta_d = {'R107':2,'R112':0,'N1115':1,'N2116':0,'N2130':2,'N1117':0,'N1118':0,'R108':0,\
		   'N1119':0,'N1120':3,'N1121':0,'N1122':1,'N1123':3,'N1124':1,'N1129':0,'N1125':0,\
		   'N1126':0,'N2129':1,'N2114':1,'N1111':2,'R101':0,'R102':2,'R104':2,'R103':0,\
		   'R105':2,'R106':2,'R108':0,'R110':0,'R112':0,'R109':2,'R111':1,'R107':2,\
		   'R2108':0,'N1110':1,'N1130':0,'N1112':1,'N1113':2,\
		   'DC04':0,'DC05':0,'DC06':0,'DC07':0,'DC08':0,'DC09':0}
S_ed = pd.Series(exsta_d)
S_ed = S_ed.sort_index()
shortlist = list(exsta_d.keys())
# shortlist = ['R2108','R102','R110','R112','1112','1115','1121','2116','R108','R107','1111','R109']
### SEEKING THE FOLLOWING RESULTS ###
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

flims = [1,6]

oI = [];oE = [];oN = [];oZ = [];oS = []

# By Elaboration, Included shaded region of raw curves
# Overlay representative curves
K_ = 0
for I_,S_ in enumerate(S_ed.index):
	if 'N' == S_[0]:
		df_m = df_M[df_M['DAS']==S_[1:]]
	else:
		df_m = df_M[df_M['DAS']==S_]
	# elabs = df_m['elaboration']
	J_ = S_ed[S_]
	E_ = df_m['elaboration'].iloc[J_]

	# for J_,E_ in enumerate(elabs):
	outs = parse_matfile(E_)

	f0_gu = df_m['f0_gau_u'].values[J_]
	ffreqs = outs['HV_ref'].index.values
	a0 = outs['HV_ref'].loc[ffreqs[np.argmin(np.abs(ffreqs-f0_gu))]]['mean']
### MAKE NEW FIGURE
	plt.figure(I_+1)
### PLOT RAW HVSR CURVES
	plt.plot(outs['HV_raw'][(outs['HV_raw'].index >= flims[0])&(outs['HV_raw'].index <= flims[1])]/a0,'k',alpha=0.1)
### PLOT MANUAL f0 PICK		
	plt.plot(df_m['f0_man'].values[J_]*np.ones(2),[0,2],'r')
### PLOT GAUSSIAN f0 ANALYSIS
	plt.plot(np.ones(2)*df_m['f0_gau_u'].values[J_],[0,2],'b--')
	plt.plot(np.ones(2)*df_m['f0_gau_u'].values[J_] - df_m['f0_gau_o'].values[J_],[0,2],'b:')
	plt.plot(np.ones(2)*df_m['f0_gau_u'].values[J_] + df_m['f0_gau_o'].values[J_],[0,2],'b:')
### PLOT REFERENCE HVSR CURVE
	plt.plot(outs['HV_ref']['mean']/a0,'w')
	plt.plot((outs['HV_ref']['mean']-outs['HV_ref']['std'])\
			 [(outs['HV_raw'].index >= flims[0])&(outs['HV_raw'].index <= flims[1])]/a0,'w:')
	plt.plot((outs['HV_ref']['mean']+outs['HV_ref']['std'])\
			 [(outs['HV_raw'].index >= flims[0])&(outs['HV_raw'].index <= flims[1])]/a0,'w:')
### FORMAT FIGURE
	plt.xlim(flims)
	plt.ylim([0,2])
	# plt.text(4,1.8,S_)
	plt.title('%d: %s (%dmE, %dmN, %dmASL)'\
			  %(I_+1,S_,df_m['mE'].values[0],df_m['mN'].values[0],df_m['mZ'].values[0]))	
	f1 = plt.gca()
	f1.axes.yaxis.set_ticks([])
		# if outs['user_f0'] is not None:
		# 	plt.plot(outs['user_f0']*np.ones(2),[0,10])
		# plt.plot(outs['auto_f0']*np.ones(2),[0,10])
### SAVE FIGURE TO SVG			
	plt.savefig('%s/%s_r%d_I%d_HVSR_v2.svg'%(OUT_DIR,S_,J_,I_+1),format='svg')
### SAVE FIGURE TO PNG
	DPI = 120
	plt.savefig('%s/%s_r%d_I%d_HVSR_v2_%ddpi.png'%(OUT_DIR,S_,J_,I_+1,DPI),format='png',dpi=DPI)
### CREATE OUTPUT INDICES
	oI.append(I_+1); oE.append(df_m['mE'].values[0]); oN.append(df_m['mN'].values[0]);
	oZ.append(df_m['mZ'].values[0]);oS.append(S_)

	K_ += 1

plt.show()

df_IDX = pd.DataFrame({'Sta':oS,'mE':oE,'mN':oN,'mZ':oZ},index=oI)
df_IDX.to_csv('%s/HVSR_Data_Index_Locs.csv'%(OUT_DIR),header=True,index=True)