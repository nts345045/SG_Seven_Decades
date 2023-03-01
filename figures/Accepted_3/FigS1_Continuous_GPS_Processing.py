import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#### MAP DATA SOURCES ####
ROOT = os.path.join('..','..','..','..')
## Raw Data File
S1A = os.path.join(ROOT,'processed_data','gps','continuous','S1A_rov1_localized.csv')
## Stitched Data File
S1B = os.path.join(ROOT,'processed_data','gps','continuous','S1C_rov1_downsampled.csv')
## Despiked Data File
S2C = os.path.join(ROOT,'processed_data','gps','continuous','S2C_rov1_despiked.csv')
## Rotated Data File
S3C = os.path.join(ROOT,'processed_data','gps','continuous','S3C_rov1_rotated.csv')
## Rolling WLS Model
S4C = os.path.join(ROOT,'processed_data','gps','continuous','S4C_rov1_vels.csv')
## Output Directory
ODIR = os.path.join(ROOT,'results','Supplement','Accepted_3')

issave = True
DPI = 300

spa = 365.24*24*3600

#### PLOTTING CONTROLS ####
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

c_cycle=['midnightblue','navy','darkblue','b','royalblue','dodgerblue','cornflowerblue','steelblue','slategray','dimgrey']


#### LOAD DATA ####
df_0 = pd.read_csv(S1A,parse_dates=True,index_col=0)
df_1 = pd.read_csv(S1B,parse_dates=True,index_col=0)
df_2 = pd.read_csv(S2C,parse_dates=True,index_col=0)
df_3 = pd.read_csv(S3C,parse_dates=True,index_col=0)
df_4 = pd.read_csv(S4C,parse_dates=True,index_col=0)
print('Data loaded')

# Calculate data-model residual
df_dx = pd.concat([df_3['mX'],df_4['x_b']],axis=1,ignore_index=False).interpolate().diff(axis=1)['x_b']
df_dx.name = 'x res'
# df_dx = df_dx[df_dx.index.isin(df_4.index)]
df_dy = pd.concat([df_3['mY'],df_4['y_b']],axis=1,ignore_index=False).interpolate().diff(axis=1)['y_b']
df_dy.name = 'y res'
# df_dy = df_dy[df_dy.index.isin(df_4.index)]
df_5 = pd.concat([df_dx,df_dy],axis=1,ignore_index=False)

ddf = [df_0,df_1,df_2,df_3,df_4**0.5,df_5]
ddfld = [('mE','mN'),('mE','mN'),('mE','mN'),('mX','mY'),('x_vmm','y_vmm'),('x res','y res'),]


fig = plt.figure(figsize=(9,10))
axs = []
isind = ['a','b','c','d','e','f','g']

for i_ in range(6):
	axs.append(plt.subplot(6,1,i_+1))
	if i_ < 3:
		axs[i_].plot(ddf[i_][ddfld[i_][0]],label=ddfld[i_][0])
		axs[i_].plot(ddf[i_][ddfld[i_][1]],label=ddfld[i_][1])
		axs[i_].set_ylabel('Disp. (m)')

	elif i_ == 3:
		axs[i_].plot(ddf[i_][ddfld[i_][0]],label=ddfld[i_][0])
		axs[i_].plot(ddf[i_][ddfld[i_][1]],label=ddfld[i_][1])
		axs[i_].set_ylabel('Disp. (m)')
		axs[i_].plot(df_4['x_b'],'k',label='mX calc')
		axs[i_].plot(df_4['y_b'],'r',label='mY calc')

	elif i_ == 4:
		axs[i_].plot(ddf[i_][ddfld[i_][0]]*spa,label='$\\sigma_{V_{surf,X}}$')
		axs[i_].plot(ddf[i_][ddfld[i_][1]]*spa,label='$\\sigma_{V_{surf,Y}}$')
		axs[i_].set_ylabel('Uncertainty (m a$^{-1}$)')
		axs[i_].set_xlabel('UTC Date Time')

	elif i_ == 5:
		axs[i_].hist(ddf[i_][ddfld[i_][0]],300,density=False,alpha=0.5,label=ddfld[i_][0])
		axs[i_].hist(ddf[i_][ddfld[i_][1]],300,density=False,alpha=0.5,label=ddfld[i_][1])
		# axs[i_].hist(df_4,300,density=False,alpha=0.5,label=ddfld[i_][0])
		# axs[i_].hist(ddf[i_][ddfld[i_][1]],300,density=False,alpha=0.5,label=ddfld[i_][1])
		
		axs[i_].set_xlabel('Residual (m)')
		axs[i_].set_xlim([-.025,.025])


	if i_ == 0:
		axs[i_].set_ylim([-2,1])
	if i_ != 4:
		axs[i_].legend(loc='upper left',ncol=2)
	else:
		axs[i_].legend(ncol=2)
	xlims = axs[i_].get_xlim()
	ylims = axs[i_].get_ylim()
	isi_ = isind[i_]
	axs[i_].text((xlims[1]-xlims[0])*0.97 + xlims[0],\
				 (ylims[1]-ylims[0])*0.8 + ylims[0],\
				 isi_,fontweight='extra bold',fontstyle='italic',fontsize=14)

plt.show()

if issave:
	save_file = os.path.join(ODIR,'SG7_Accepted_FigS1_Continuous_GPS_Processing_%ddpi.png'%(DPI))
	plt.savefig(save_file,dpi=DPI,format='PNG')