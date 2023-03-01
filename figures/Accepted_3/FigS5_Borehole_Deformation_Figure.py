import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#### MAP DATA SOURCES ####
ROOT = os.path.join('..','..','..','..')
## Cleaned Borehole Data
BHD = os.path.join(ROOT,'processed_data','deformation','Processed_Borehole_Deformation_Data.csv')
## LOOCV Results
BHM = os.path.join(ROOT,'processed_data','deformation','Effective_Viscosity_LOOCV_results.csv')
## Output Directory
ODIR = os.path.join(ROOT,'results','Supplement','Accepted_3')

issave = True
DPI = 300

### Forward Equations
# Nonlinear Viscous Flowlaw
def nlv(Td,B):
	return (Td/B)**3.
# Define Driving Stress Function
def Td(rho,g,z,alpha):
	return rho*g*z*np.sin(alpha)

### Physical Parameters
alpha = 4.41 * (np.pi/180)
rho = 910
gg = 9.81
z_vect = np.linspace(1,45,100)
Td_ = Td(rho,gg,z_vect,alpha)
#### PLOTTING CONTROLS ####
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
c_cycle=['midnightblue','navy','darkblue','b','royalblue','dodgerblue','cornflowerblue','steelblue','slategray','dimgrey']


#### LOAD DATA ####
df_BHD = pd.read_csv(BHD,index_col=0)
df_BHM = pd.read_csv(BHM)

eddm = df_BHD['edot_mea(a-1)'] - df_BHD['edot_min(a-1)']
eddM = df_BHD['edot_max(a-1)'] - df_BHD['edot_mea(a-1)']

fig,ax2 = plt.subplots(figsize=(7.9,7.9))
ax1 = ax2.twinx()

handles = []
for i_ in range(len(df_BHM)):
	ci_ = c_cycle[i_]
	LVi = df_BHM.loc[i_,'LOOCV']
	if i_ == 0:
		LVi = 'None'
	Bu = df_BHM.loc[i_,'B(Pa a1/3)']
	Bs = df_BHM.loc[i_]['B_var(Pa2 a2/3)']**0.5
	edh_u = nlv(Td_,Bu)
	edh_su = nlv(Td_,Bu + Bs)
	edh_sl = nlv(Td_,Bu - Bs)
	if LVi != 'None':
		Ostr = str(int(LVi+1))
	else:
		Ostr = str(LVi)
	h_, = ax2.plot(edh_u,Td_*1e-3,color=ci_,\
				   label='B: %.3f$\\pm$%.3f MPa a$^{1/3}$ O:%s'%\
				   (Bu*1e-6,Bs*1e-6,Ostr))
	handles.append(h_)

	## Plot Model Fit and Uncertainties ##
	ax2.fill_betweenx(Td_*1e-3,edh_sl,edh_su,color=ci_,alpha=0.25)

	## Plot Labels On Data ##
	if i_ > 0:
		ax1.text(df_BHD['edot_mea(a-1)'].values[i_-1] + 0.0001,df_BHD.index[i_-1] - 1,\
				 i_,fontsize=16)

## Plot Measurements from Meier (1957) ##
h_ = ax1.errorbar(df_BHD['edot_mea(a-1)'],df_BHD.index,xerr=np.array([eddm.values,eddM.values]),\
			 label='Data',fmt='o',color='k',capsize=5)
ax1.legend(handles=[h_]+handles,loc='upper right')

ax1.set_ylim([0,45])
ax1.invert_yaxis()
ax2.set_ylim([0,Td(rho,gg,45,alpha)*1e-3])
ax2.invert_yaxis()
ax1.set_xlim([-0.001,0.006])

ax2.set_ylabel('Driving Stress $\\tau_{xz}$ (kPa)')
ax1.set_ylabel('Depth Below Ground Surface (m b.g.s.)',rotation=270,labelpad=15)
ax2.set_xlabel('Shear Strain Rate $\\dot{\\epsilon}_{xz}$ (a$^{-1}$)')

plt.show()

if issave:
	save_file = os.path.join(ODIR,'SG7_Accepted_FigS5_Borehole_Deformation_%ddpi.png'%(DPI))
	plt.savefig(save_file,dpi=DPI,format='PNG')