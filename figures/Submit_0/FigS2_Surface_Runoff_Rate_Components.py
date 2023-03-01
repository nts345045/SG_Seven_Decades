import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

### Saving Controls ###
issave = True
FMT = 'png'
DPI = 200
PIN = 0.05
### Data & Metadata Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..')
# m_f estimate data file (csv format)
IMDATA = os.path.join(DROOT,'processed_data','runoff','melt_factor_estimates.csv')
# Summary Data File (csv format)
ISDATA = os.path.join(DROOT,'processed_data','runoff','Surface_Runoff_Rates.csv')


df_mf = pd.read_csv(IMDATA,index_col='midx',parse_dates=True)
df_drsm = pd.read_csv(ISDATA,index_col=0,parse_dates=True)

fig = plt.figure(figsize=[12,7],constrained_layout=True)
gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0,:-1])
ax1.plot(df_drsm['TEMP(C)'],'k-')
ax1.set_ylabel('Air Temperature\nat AWS ($^\\circ C$)')
ax1.set_ylim([0,18])
# ax1.xaxis.set_visible(False)
ax1.set_title('A')

ax2 = fig.add_subplot(gs[1,:-1],sharex=ax1)
ax2.plot(df_drsm['Rdot_rain(mm/hr)'],'k-',label='$\\dot{R}_{rain}$')
ax2.plot(df_drsm['Mdot(mmWE/hr)'],'r-',label='$\\dot{M}$')
ax2.plot(df_drsm['Rdot_surf(mmWE/hr)'],'b-.',label='$\\dot{R}_{surf}$')
ax2.set_ylabel('Runoff Rate\n($mm_{WE}$ $hr^{-1}$)')
ax2.set_xlabel('Local Time (UTC-7)')
ax2.legend(ncol=3)
ax2.set_title('B')

ax3 = fig.add_subplot(gs[:,-1])
_,bins,_=ax3.hist(df_mf['MF'],50,color='k',density=True,label='Individual Estimates')
u,o = norm.fit(df_mf['MF'].values)
y = norm.pdf(bins,u,o)
ax3.plot(bins,y,'r-')
ax3.text(0,7,'Fit: $%.2f\\pm%.2f mm_{WE}$$^\\circ C^{-1} hr^{-1}$'%(u,o))
ax3.set_title('C')
ax3.set_xlabel('Melt-Factor [$m_f$]\n($mm_{WE}$$^\\circ C^{-1} hr^{-1}$)')
ax3.set_ylim([0,7.25])
ax3.set_ylabel('Frequency')


plt.show()

if issave:
	plt.savefig(os.path.join(DROOT,'results','Supplement','FigS2_Surface_Runoff_Processing_%ddpi.%s'%(DPI,FMT.lower())),dpi=DPI,pad_inches=PIN,format=FMT.lower())
