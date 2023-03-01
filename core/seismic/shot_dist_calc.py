import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

root = '/home/nates/ActiveProjects/SGGS/ANALYSIS/REFRACTION'
geom_file = 'FIELD_NOTES/Shot_Geometries_UTM11N.csv'
out_file = 'PROCESSED/Associated_Shot_Times.csv'
shot_geom = os.path.join(root,geom_file)
print(shot_geom)
time_marks = glob(os.path.join(src,'MARKERS/S0??_Stack_Picks'))
time_marks.sort()

# Compose Reduced Shot-Time DataFrame
time_df = pd.DataFrame() 
for pf in time_marks: 
	plist = pm.load_markers(pf) 
	ishot = [] 
	shot_name = pf.split('/')[-1].split('_')[0] 
	idt = [] 
	iterr = []
	iphz = [] 
	iloc = [] 
	for p_ in plist: 
		if isinstance(p_,pm.EventMarker): 
			t0 = p_.tmin 
		elif isinstance(p_,pm.PhaseMarker): 
			pz = p_.get_attributes()[-3]
			idt.append(p_.tmin) 
			if p_.tmax - p_.tmin < 0.001 and pz != 'R':
				iterr.append(0.001)
			elif p_.tmax - p_.tmin < 0.001 and pz == 'R':
				iterr.append(0.0025)
			else:
				iterr.append(p_.tmax - p_.tmin)
			iphz.append(p_.get_attributes()[-3]) 
			iloc.append('.'.join(list(p_.get_nslc_ids()[0]))) 
			ishot.append(shot_name) 
	idt = np.array(idt) 
	idt -= t0 
	itime_df = pd.DataFrame({'shot':ishot,'dt(s)':idt,'err(s)':iterr,'phz':iphz,'nslc':iloc},index=iloc) 
	time_df = pd.concat([time_df,itime_df],ignore_index=False) 

# Load Shot Geometry Data
shot_df = pd.read_csv(shot_geom)


# Create holder lists for key outputs of association
otimes = []
oterrs = []
odists = []
oshots = []
onslcs = []
ophzs = []

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
		otimes.append(time_df[time_df['shot']==shot]['dt(s)'].values[i_])
		oterrs.append(time_df[time_df['shot']==shot]['err(s)'].values[i_])
		ophzs.append(time_df[time_df['shot']==shot]['phz'].values[i_])

# Compile Results into DataTable
dtdx_df = pd.DataFrame({'nslc':onslcs,'shot':oshots,'d(m)':odists,'dt(s)':otimes,'err(s)':oterrs,'phz':ophzs},index=onslcs)
# Save dt/dx association data
dtdx_df.to_csv(os.path.join(root,out_file),header=True,index=False)

df_surf = dtdx_df[dtdx_df['phz']=='R']
#### SLOPE MODELLING SECTION ####
# Conduct a 1st degree weighted least squares poly-fit
Rpoly = np.polyfit(df_surf['d(m)'].values,df_surf['dt(s)'].values,1,\
				   w=df_surf['err(s)']**-2)
Rfun = np.poly1d(Rpoly)
print('Surface Wave Velocity: %.3f km/s'%((Rpoly[0]**-1)/1000.))


# Subset Direct P-phases (based on visual inspection - plotted below)
df_P = dtdx_df[dtdx_df['phz']=='P']
df_P = df_P[df_P['d(m)'] <= 220]
# Conduct a 1st degree weighted least squares poly-fit
Ppoly = np.polyfit(df_P['d(m)'].values,df_P['dt(s)'].values,1,\
					w=df_P['err(s)']**-2)
Pfun = np.poly1d(Ppoly)
print('Direct P-Wave Velocity: %.3f km/s'%((Ppoly[0]**-1)/1000.))

# Subset Refracted P-Phases (based on visual inspection - plotted below)
df_ref = dtdx_df[dtdx_df['phz'] != 'R']
df_ref = df_ref[df_ref['d(m)'] >= 300]
# Conduct a 1st degree weighted least squares poly-fit
ref_poly = np.polyfit(df_ref['d(m)'].values,df_ref['dt(s)'].values,1,\
					w=df_ref['err(s)']**-2)
ref_fun = np.poly1d(ref_poly)
print('Refracted P-Wave Velocity: %.3f km/s'%((ref_poly[0]**-1)/1000.))

# Investigate Reflected P-wave Data
df_Pp = dtdx_df[dtdx_df['phz']!='R']
df_Pp = df_Pp[df_Pp['d(m)'] > 220]
df_Pp = df_Pp[df_Pp['d(m)'] < 300]
Pp_poly = np.polyfit(df_Pp['d(m)'].values,df_Pp['dt(s)'].values,1,\
					w=df_Pp['err(s)']**-2)
Ppfun = np.poly1d(Pp_poly)
print('Reflected P-Wave Velocity: %.3f km/s'%((Pp_poly[0]**-1)/1000.))

plt.figure()
plt.errorbar(df_surf['d(m)'],df_surf['dt(s)'],yerr=df_surf['err(s)'],fmt='k.',capsize=5)
plt.errorbar(df_P['d(m)'],df_P['dt(s)'],yerr=df_P['err(s)'],fmt='b.',capsize=5)
plt.errorbar(df_ref['d(m)'],df_ref['dt(s)'],yerr=df_ref['err(s)'],fmt='r.',capsize=5)
plt.errorbar(df_Pp['d(m)'],df_Pp['dt(s)'],yerr=df_Pp['err(s)'],fmt='m.',capsize=5)
xvec = dtdx_df['d(m)'].unique()
plt.plot(xvec,Rfun(xvec),'k--')
plt.plot(xvec,Pfun(xvec),'b--')
plt.plot(xvec,ref_fun(xvec),'r--')
plt.plot(xvec,Ppfun(xvec),'m--')


#### PLOT RESIDUAL DISTRIBUTIONS ### (REALLY SHOULD DO Q-Q PLOTS...)
plt.figure()
plt.subplot(221)
plt.hist(df_surf['dt(s)'].values - Rfun(df_surf['d(m)']),20)
xlabel('Surface Phase Residual (s)')
plt.subplot(222)
plt.hist(df_P['dt(s)'].values - Pfun(df_P['d(m)']),20)
xlabel('Direct P Residual (s)')
plt.subplot(223)
plt.hist(df_ref['dt(s)'].values - ref_fun(df_ref['d(m)']))
xlabel('Refracted P Residual (s)')
plt.subplot(224)
plt.hist(df_Pp['dt(s)'].values - Ppfun(df_Pp['d(m)']))
xlabel('Reflected P Residual (s)')

###### PLOTTING SECTION ######

## Display dt/dx data and color by shot number
plt.figure()
for shot in dtdx_df['shot'].unique():
	plt.plot(dtdx_df[dtdx_df['shot']==shot]['d(m)'].values,\
				dtdx_df[dtdx_df['shot']==shot]['dt(s)'].values,'.',label=shot)
plt.ylabel('Travel Time (s)')
plt.xlabel('Offset (m)')
plt.legend()
## Display dt/dx data and color by phase
plt.figure()
for phz in dtdx_df['phz'].unique():
	plt.plot(dtdx_df[dtdx_df['phz']==phz]['d(m)'].values,\
				dtdx_df[dtdx_df['phz']==phz]['dt(s)'].values,'.',label=phz)
plt.ylabel('Travel Time (s)')
plt.xlabel('Offset (m)')
plt.legend()
## Display dt/dx data and color by station
plt.figure()
for nslc in dtdx_df['nslc'].unique():
	if nslc == '.c0AU4..p0':
		plt.plot(dtdx_df[dtdx_df['nslc']==nslc]['d(m)'].values,\
				dtdx_df[dtdx_df['nslc']==nslc]['dt(s)'].values,'ks',label=nslc)
	else:
		plt.plot(dtdx_df[dtdx_df['nslc']==nslc]['d(m)'].values,\
				dtdx_df[dtdx_df['nslc']==nslc]['dt(s)'].values,'.',label=nslc)
plt.ylabel('Travel Time (s)')
plt.xlabel('Offset (m)')
plt.legend()