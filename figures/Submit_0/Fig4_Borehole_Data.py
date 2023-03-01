from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

#### PHYSICS ####
# Physical Parameters
ft2m = 1./3.2808 			# [ft/m] Length conversion factor
yr2sec = 365.24*24.*3600.   # [sec/year] Time conversion factor
phi = 916.7					# [kg/m**3] Glacier ice density
gg = 9.81 					# [m/s**2] Gravitational Acceleration
aa = 4.41*(3.141592/180.)	# [rad] Surface Slope @ Meier (1957) Borehole

# Define NonLinear Viscous Rheology Function
def nlv(Td,B):
	return (Td/B)**3.
# Define Driving Stress Function
def Td(rho,g,z,alpha):
	return rho*g*z*np.sin(alpha)


#### PLOTTING CONTROLS ####
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#### LOAD DATA ####
# File Path Definitions
ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/FLOW_MODELS/B_Inversion'
# Load Data & Convert to SI
df_edot_m = pd.read_csv(ROOT+'/Meier_1957/Mean_Strain_Rate.csv',index_col='z(ft)')
df_edot_m.index= df_edot_m.index*ft2m
df_edot_m.index.name='z(m)'
df_edot_l = pd.read_csv(ROOT+'/Meier_1957/SR_Lower_Bound.csv',index_col='z(ft)')
df_edot_l.index= df_edot_m.index*ft2m
df_edot_l.index.name='z(m)'
df_edot_u = pd.read_csv(ROOT+'/Meier_1957/SR_Upper_Bound.csv',index_col='z(ft)')
df_edot_u.index= df_edot_m.index*ft2m
df_edot_u.index.name='z(m)'
# Sort for increasing depth
df_edot_m = df_edot_m.sort_index()
df_edot_l = df_edot_l.sort_index()
df_edot_u = df_edot_u.sort_index()
# Extract epsilon-dot values
ed = df_edot_m.values[:,0]
# Get average error for each depth-point
ed_sigma = (df_edot_u.values - df_edot_l.values)/(2.*1.96)
ed_sigma = ed_sigma[:,0]
# Model Driving Stress
Td = Td(phi,gg,df_edot_m.index.values,aa)


#### INVERSION SECTION ####
# Invert for B and covB
popt,pcov = curve_fit(nlv,Td,ed,sigma=ed_sigma)#,sigma=ed_sigma)
# Calculate L2 norm of residual
r_l2 = np.linalg.norm(nlv(Td,popt[0]) - ed)
# Generate full-data inclusion lines for model outputs
B_est = [popt[0]]; B_std = [pcov[0][0]**0.5]; oI = ['None']; oZ = ['None']; rL2 = [r_l2]


#### LOOCV PROCESSING ####
# Prepare Leave-One-Out Cross Validation Indices
ind = np.arange(len(df_edot_m))
loo = LeaveOneOut()
loo.get_n_splits(ind)
# Do LOOCV Processing
for t_idx, o_idx in loo.split(ind):
	ipopt,ipcov = curve_fit(nlv,Td[t_idx],ed[t_idx],sigma=ed_sigma[t_idx])
	r_l2 = np.linalg.norm(nlv(Td[t_idx],ipopt[0]) - ed[t_idx])
	oI.append(o_idx[0])
	oZ.append(df_edot_m.index.values[o_idx[0]])
	B_est.append(ipopt[0])
	B_std.append(ipcov[0][0]**0.5)
	rL2.append(r_l2)

# Write results to DataFrame
df_B = pd.DataFrame({'LOOCV':oI,'LOOCV_z':oZ,'B':B_est,'std':B_std,'L2':rL2})
# display(df_B)


#### PLOTTING SECTION ####
# Display Results
fig,ax = plt.subplots()
ax.errorbar(ed,df_edot_m.index.values,xerr=1.96*ed_sigma,capsize=5,label='Data',fmt='o')
ax2 = ax.twinx()
for i_ in range(len(df_B)):
	iS = df_B.iloc[i_]
	B_ = iS['B']
	oB_ = iS['std']
	r2_ = iS['L2']
	ax2.plot(nlv(Td,B_),Td*1e-3,label='B: %.3f $\pm$ %.3f $MPa/a^{1/3}$ Omitted: %s'%(B_/1e6,2*oB_/1e6,str(iS['LOOCV'])))
	ax2.fill_betweenx(Td*1e-3,nlv(Td,B_- 1.96*oB_),nlv(Td,B_ + 1.96*oB_),alpha=0.2)
	if i_ < len(df_B) - 1:
		ax2.text(ed[i_],Td[i_]*1e-3,i_)

# ax.legend()
ax2.legend()
ax.set_xlabel('Strain Rate ($\dot{\epsilon}$) [1/yr]')
ax.invert_yaxis()
ax.set_ylabel('Depth (z) [m]')
ax2.invert_yaxis()
ax2.set_ylabel('Driving Stress ($\\tau_b$) [kPa]')

plt.show()

#### SAVE RESULTS ####
df_B.to_csv(ROOT+'/Inversion_Results_LOOCV.csv',header=True,index=False)