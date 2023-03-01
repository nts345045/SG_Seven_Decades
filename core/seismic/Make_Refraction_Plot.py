import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrocko.gui.marker as pm
import sys
import os
from glob import glob
from obspy import read
sys.path.append('/home/nates/Scripts/Python/PKG_DEV')
# from 
def S2V(Slowness,SlowVar):
    v = Slowness**-1
    sp = Slowness + SlowVar**0.5
    sm = Slowness - SlowVar**0.5
    vp = np.abs(v - sp**-1)
    vm = np.abs(v - sm**-1)
    vvar = np.max([vp,vm])**2
    return v, vvar

# Define Directory Paths
IROOT = '/home/nates/ActiveProjects/SGGS/ANALYSIS/REFRACTION'
# Receiver/Source Geometry File
geom_file = IROOT+'/FIELD_NOTES/Shot_Geometries_UTM11N.csv'
# List of Common Shot Gather Pick Times
pick_files = glob(IROOT+'/MARKERS/S0??_Stack_Picks')
# Get a list of mean stacks that have been processed
stack_files = list(np.sort(glob(IROOT+'/STACKED_SECTIONS/S*_1kHz_mean_stack.mseed')))
# distance / delta-time output file
out_file = os.path.join(IROOT,'PROCESSED/Associated_Shot_Times.csv')


# Define Minimum Pick Error Floor
errPmin = 0.001
errQmin = 0.005
errRmin = 0.0025

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# Compose Reduced Time DataFrame from Common Shot Gather Arrival Times
time_df = pd.DataFrame() 
for pf in pick_files: 
    plist = pm.load_markers(pf) 
    ishot = [] 
    shot_name = pf.split('/')[-1].split('_')[0] 
    ist = []
    idt = [] 
    iterr = []
    iphz = [] 
    iloc = [] 
    for p_ in plist: 
        if isinstance(p_,pm.EventMarker): 
            t0 = p_.tmin 
            ist = t0
        elif isinstance(p_,pm.PhaseMarker): 
            pz = p_.get_attributes()[-3]
            idt.append(np.mean([p_.tmin,p_.tmax]))
            # Apply Minimum Pick Uncertainty Errors
            if p_.tmax - p_.tmin < errPmin and pz == 'P':
                iterr.append(errPmin)
            elif p_.tmax - p_.tmin < errRmin and pz == 'R':
                iterr.append(errRmin)
            elif p_.tmax - p_.tmin < errQmin and pz == '?':
                iterr.append(errQmin)
            else:
                iterr.append(p_.tmax - p_.tmin)
            iphz.append(p_.get_attributes()[-3]) 
            iloc.append('.'.join(list(p_.get_nslc_ids()[0]))) 
            ishot.append(shot_name) 
    idt = np.array(idt) 
    idt -= t0 
    itime_df = pd.DataFrame({'shot':ishot,'t0':t0,'dt(s)':idt,'err(s)':iterr,'phz':iphz,'nslc':iloc},index=iloc) 
    time_df = pd.concat([time_df,itime_df],ignore_index=False) 

# Load Shot Geometries
shot_df = pd.read_csv(geom_file)

### Compose Distance Matrices and Associate to Common Shot Gathers
# Create holder lists for key outputs of association
otimes = []; oterrs = []; odists = []
oshots = []; onslcs = []; ophzs = []
t0s = []
## Iterate Across Recognized Shot Counts
shot_list = time_df['shot'].unique()
for shot in shot_list:
	# Calculate shot-specific distances
	shot_dels = shot_df[shot_df[shot]>0][['mE','mN','Elev']].values - \
				 shot_df[shot_df[shot]==2][['mE','mN','Elev']].values
	shot_dists = np.sqrt(np.sum(shot_dels**2,axis=1))
	dshot_df = pd.DataFrame({'d(m)':shot_dists},\
							 index=shot_df[shot_df[shot]>0]['nslc'])
	# Iterate across stations picked from shot gather
	for i_,nslc in enumerate(time_df[time_df['shot']==shot]['nslc']):
		idist = dshot_df[dshot_df.index==nslc]['d(m)'].values[0]
		onslcs.append(nslc)
		oshots.append(shot)
		odists.append(idist)
		tidx = time_df['shot'] == shot
		otimes.append(time_df[tidx]['dt(s)'].values[i_])
		oterrs.append(time_df[tidx]['err(s)'].values[i_])
		ophzs.append(time_df[tidx]['phz'].values[i_])
		t0s.append(time_df[tidx]['t0'].values[i_])
# Compile Results into DataTable
dtdx_df = pd.DataFrame({'nslc':onslcs,'shot':oshots,'d(m)':odists,\
                        'dt(s)':otimes,'err(s)':oterrs,'phz':ophzs,\
                        't0(s)':t0s},index=onslcs)

# Plot Example Stack with Picks
	# ishot = 'S001'
norm_dx = 100
clip_dx = 25
ppick_lim = 220
Q_ = 25
cind = {'P':'firebrick','R':'dodgerblue','?':'purple'}
plt.figure()
slist = ['S009']#dtdx_df['shot'].unique();#['S002']#,'S003','S009','S013']
# Get referencing index
IDX = dtdx_df['shot'].isin(slist)
dfi_p = dtdx_df[IDX & (dtdx_df['phz']=='P')&(dtdx_df['d(m)']<=ppick_lim)]
# Fit Model for V_P
m_p,c_p = np.polyfit(dfi_p['d(m)'].values,dfi_p['dt(s)'].values,1,w=dfi_p['err(s)']**-1,cov=True)
Vpu,Vps = S2V(m_p[0],c_p[0][0])
# Fit Model for V_R
dfi_r = dtdx_df[IDX & (dtdx_df['phz']=='R')]
m_r,c_r = np.polyfit(dfi_r['d(m)'].values,dfi_r['dt(s)'].values,1,w=dfi_r['err(s)']**-1,cov=True)
Vru,Vrs = S2V(m_r[0],c_r[0][0])

d_v = np.linspace(0,600)

plt.plot(d_v[d_v <= ppick_lim],np.poly1d(m_p)(d_v[d_v <= ppick_lim])*1000,\
		 '-',color=cind['P'],label='$V_{P}:%.1f \pm %.1f m/s$'%(Vpu,Vps**0.5))
plt.plot(d_v[d_v >= ppick_lim],np.poly1d(m_p)(d_v[d_v >= ppick_lim])*1000,'--',color=cind['P'])#,label='$V_{P}:%.1f \pm %.1f m/s$'%(Vpu,Vps**0.5))

plt.plot(d_v,np.poly1d(m_r)(d_v)*1000,'-',color=cind['R'],label='$V_{R}:%.1f \pm %.1f m/s$'%(Vru,Vrs**0.5))

plt.legend()

for ishot in slist:

	# Subset to shot summary dataframe
	dfi = dtdx_df[dtdx_df['shot']==ishot]
	dfi = dfi.sort_values('d(m)')
	# Load mean stack 
	st_stack = read(IROOT+'/STACKED_SECTIONS/%s_1kHz_mean_stack.mseed'%(ishot))
	# 
	# plt.figure()
	for i_,id_ in enumerate(dfi.index.unique()):
		# Pull indexed trace data
		tr_i = st_stack.select(id=id_)[0].copy().normalize()
		# Get distance
		sdfi = dfi.loc[id_]
		if isinstance(sdfi,pd.DataFrame):
			d_ = dfi.loc[id_]['d(m)'].values[0]
		else:
			d_ = dfi.loc[id_]['d(m)']
		# c_max = tr_i.data.max()
		# c_min = tr_i.data.min()
		# c_norm = 0.5*(c_max + c_min)
		# Get trigger time
		t0 = dfi['t0(s)'].values[i_]
		dt = int(tr_i.stats.sampling_rate)**-1
		# t_v = np.arange(tr_i.stats.starttime.timestamp,tr_i.stats.endtime.timestamp,dt) - t0
		t_v = np.linspace(tr_i.stats.starttime.timestamp,tr_i.stats.endtime.timestamp,tr_i.stats.npts) - t0
		# Scale Normalization (GAIN STATEMENT)
		d_v = tr_i.data*norm_dx*(1 - d_/650)*-1
		# d_v = tr_i.data*norm_dx*Q_**-1*(d_/1e3)**3
		# Conduct clipping
		for i_ in range(len(d_v)):
			if np.abs(d_v[i_]) >= clip_dx:
				d_v[i_] = clip_dx*np.sign(d_v[i_])
		# Plot Trace
		plt.plot(d_v + d_, t_v*1000,'k',alpha=0.5)
		plt.fill_betweenx(t_v*1000,d_v + d_,np.zeros(t_v.shape) + d_,color='k',alpha=0.1)
		# plt.fill_betweenx(t_v*1000,tr_i.data*norm_dx + d_,np.zeros(t_v.shape) + d_,color='r',alpha=0.5)
		for j_ in range(len(sdfi)):
			if isinstance(sdfi,pd.DataFrame):
				sdt = sdfi.iloc[j_]['dt(s)']*1000
				spz = sdfi.iloc[j_]['phz']
			else:
				sdt = sdfi['dt(s)']*1000
				spz = sdfi['phz']
			if spz == 'R':
				plt.plot([-5,5] + d_,np.ones(2)*sdt,color=cind[spz],alpha=0.75)
			elif spz == 'P' and d_ <= ppick_lim:
				plt.plot([-5,5] + d_,np.ones(2)*sdt,color=cind[spz],alpha=0.75)
plt.gca().invert_yaxis()
plt.title('Shot: %s'%(ishot))
plt.xlabel('3-D Source-Receiver Offset (m)')
plt.ylabel('Travel Time (ms)')

plt.show()

# norm_dx = 25
# cind = {'P':'firebrick','R':'dodgerblue','?':'purple'}
# plt.figure()
# for ishot in dtdx_df['shot'].unique():

# 	# Subset to shot summary dataframe
# 	dfi = dtdx_df[dtdx_df['shot']==ishot]
# 	dfi = dfi.sort_values('d(m)')
# 	# Load mean stack 
# 	st_stack = read(IROOT+'/STACKED_SECTIONS/%s_1kHz_mean_stack.mseed'%(ishot))
# 	# 
# 	# plt.figure()
# 	for i_,id_ in enumerate(dfi.index.unique()):
# 		# Pull indexed trace data
# 		tr_i = st_stack.select(id=id_)[0].copy().normalize()
# 		# Get distance
# 		sdfi = dfi.loc[id_]
# 		if isinstance(sdfi,pd.DataFrame):
# 			d_ = dfi.loc[id_]['d(m)'].values[0]
# 		else:
# 			d_ = dfi.loc[id_]['d(m)']
# 		# c_max = tr_i.data.max()
# 		# c_min = tr_i.data.min()
# 		# c_norm = 0.5*(c_max + c_min)
# 		# Get trigger time
# 		t0 = dfi['t0(s)'].values[i_]
# 		dt = int(tr_i.stats.sampling_rate)**-1
# 		# t_v = np.arange(tr_i.stats.starttime.timestamp,tr_i.stats.endtime.timestamp,dt) - t0
# 		t_v = np.linspace(tr_i.stats.starttime.timestamp,tr_i.stats.endtime.timestamp,tr_i.stats.npts) - t0
# 		plt.plot(tr_i.data*norm_dx + d_, t_v*1000,'k',alpha=0.1)
# 		# plt.fill_betweenx(t_v*1000,tr_i.data*norm_dx + d_,np.zeros(t_v.shape) + d_,color='r',alpha=0.5)
# 		for j_ in range(len(sdfi)):
# 			if isinstance(sdfi,pd.DataFrame):
# 				sdt = sdfi.iloc[j_]['dt(s)']*1000
# 				spz = sdfi.iloc[j_]['phz']
# 			else:
# 				sdt = sdfi['dt(s)']*1000
# 				spz = sdfi['phz']
# 			# plt.plot([-10,10] + d_,np.ones(2)*sdt,color=cind[spz],alpha=0.25)
# 	plt.gca().invert_yaxis()
# 	plt.title('Shot: %s'%(ishot))
# 	plt.xlabel('3-D Source-Receiver Offset (m)')
# 	plt.ylabel('Travel Time (ms)')