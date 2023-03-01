import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def remove_norm(df_MT,df_MN,Yr1=1981):
	df_NT = df_MT.copy()
	for m_ in df_MT['Month'].unique():
		cond = df_NT['Month'] == m_
		df_NT.loc[cond,'Temp(C)'] -= df_MN[df_MN['Yr1']==Yr1].loc['mean',m_]

	return df_NT

def average_temps(df_MT,months1,months2=[],sameyear=True):
	if sameyear:
		months = months1 + months2
	IDF = df_MT.copy()
	yrs = []
	pyrs = []
	temps = []
	nans = []
	cnans = []
	for y_ in IDF['Year'].unique():
		if sameyear:
			idf = IDF[(IDF['Year']==y_) & (IDF['Month'].isin(months))]
		else:
			cond1 = (IDF['Year'] == y_) & (IDF['Month'].isin(months1))
			cond2 = (IDF['Year'] == y_ + 1) & (IDF['Month'].isin(months2))
			idf = IDF[cond1 | cond2]
		# Get total number of nans
		n_nan = np.sum(idf['Temp(C)'].isna().values)
		# Get maximum number of consecutive nans
		b = idf['Temp(C)'].isna().cumsum()
		c_na = b.max()
		# Enforce 3 and 5 rule
		if n_nan <= 5 and c_na < 3:
			T_bar = idf['Temp(C)'].mean()
			temps.append(T_bar)
		else:
			temps.append(np.nan)
		# Append results
		nans.append(n_nan)
		cnans.append(c_na)
		yrs.append(y_)
		if sameyear:
			pyrs.append(pd.Timestamp(str(y_)) + pd.Timedelta(365,unit='day')/2)	
		else:
			pyrs.append(pd.Timestamp(str(y_+1)))
	df_NAT = pd.DataFrame({'Year':yrs,'Temp(C)':temps,'Tnans':nans,'Cnans':cnans},index=pyrs)

	return df_NAT

def normalize_and_average(df_MT,df_MN,months1,months2=[],sameyear=True,Yr1=1981):
	"""
	Create a temperature anomaly curve from monthly temperature data and
	a climate normal and then create an average for some specified set of months.

	:: INPUTS ::
	:type df_MT: pandas.DataFrame
	:param df_MT: temperature dataframe with years and months as other columns
	:type df_MN: pandas.DataFrame
	:param df_MN: temperature normals data frame with columns as months
	:type months1: list
	:param months1: list of string months to use for averaging temperatures
	:type months2: optional list
	:param months2: list of string months to use from the subsequent year
					allowing year-wrapping for winter/accumulation season
					statistics
	:type sameyear: bool
	:param sameyear: are months1 and months2 in the same year, or is 
					months2 in the next year?
	:type Yr1: int
	:param Yr1: initial year to use for getting the correct climate normals entry

	:: OUTPUTS ::
	:rtype df_NT: pandas.DataFrame
	:return df_NT: temperature anomaly data (observed - normal), same format as df_MT
	:rtype df_AT: pandas.DataFrame
	:return df_AT: averaged temperatures
	:rtype df_NAT: pandas.DataFrame
	:return df_NAT: averaged temperature anomaly data

	"""
	# Remove climate normal from data
	df_NT = remove_norm(df_MT,df_MN,Yr1=Yr1)
	df_AT = average_temps(df_MT,months1,months2=months2,sameyear=sameyear)
	df_NAT = average_temps(df_NT,months1,months2=months2,sameyear=sameyear)
	return df_NT,df_AT,df_NAT


#################################################################################
#### SET DATA PATHS #############################################################
#################################################################################
# Project Directory
ROOT = os.path.join('..','..','..','..')
## CLIMATE DATA
# Climate Data Directory
CROOT = os.path.join(ROOT,'data','MINDSatUW','CLIMATE')
# Climate Normals for Nordegg, AB Climate Reference Station
NORMN = os.path.join(CROOT,'Temperature','Nordegg_Climate_Normals.csv')
# Climate Normals for Nordegg, AB Climate Reference Station
NORMG = os.path.join(CROOT,'Temperature','Golden_Climate_Normals.csv')
# Climate Normals for Nordegg, AB Climate Reference Station
NORMJ = os.path.join(CROOT,'Temperature','Jasper_Climate_Normals.csv')

# Homogenized Monthly Temperatures for Nordegg, AB Climate Reference Station
HOMTN = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'NORDEGG','NORDEGG_mean_monthly_stack.csv')
# Homogenized Monthly Temperatures for Golden, AB Climate Reference Station
HOMTG = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'GOLDEN','GOLDEN_mean_monthly_stack.csv')
# Homogenized Monthly Temperatures for Jasper, AB Climate Reference Station
HOMTJ = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'JASPER','JASPER_mean_monthly_stack.csv')

# Homogenized Seasonal Temperatures for Nordegg, AB Climate Reference Station
HOSTN = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'NORDEGG','NORDEGG_mean_season_stack.csv')
# Homogenized Seasonal Temperatures for Golden, AB Climate Reference Station
HOSTG = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'GOLDEN','GOLDEN_mean_season_stack.csv')
# Homogenized Seasonal Temperatures for Jasper, AB Climate Reference Station
HOSTJ = os.path.join(CROOT,'Temperature','Extracted_Sites',\
								  'JASPER','JASPER_mean_season_stack.csv')
# Figure output directory
ODIR = os.path.join(ROOT,'results','Main','Accepted_3')
issave = True
DPI = 300
### Weather & Climate Data ###
# Load Homogenized Temperatures for Nordegg Climate Reference Station
df_HOMTN = pd.read_csv(HOMTN,parse_dates=True,index_col=0)
df_HOMTG = pd.read_csv(HOMTG,parse_dates=True,index_col=0)
df_HOMTJ = pd.read_csv(HOMTJ,parse_dates=True,index_col=0)


df_HOSTN = pd.read_csv(HOSTN,parse_dates=True,index_col=0)
df_HOSTG = pd.read_csv(HOSTG,parse_dates=True,index_col=0)
df_HOSTJ = pd.read_csv(HOSTJ,parse_dates=True,index_col=0)

# Load Climate Norms for Nordegg Climate Reference Station
df_NORMN = pd.read_csv(NORMN,index_col=0)
df_NORMG = pd.read_csv(NORMG,index_col=0)
df_NORMJ = pd.read_csv(NORMJ,index_col=0)


# Months Index for easy pull of multi-indexed normals data
mlist = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mslist = ['Apr','May','Jun','Jul','Aug','Sep',]
aslist = ['Jan','Feb','Mar','Oct','Nov','Dec']

# Normalize Monthly Data & Get Annual, Melt Season, and Accumulation Season Temperature Averages
df_NdT,df_NA,df_NdTA = normalize_and_average(df_HOMTN,df_NORMN,mlist)
_,df_Na,df_NdTa = normalize_and_average(df_HOMTN,df_NORMN,aslist[:3],months2=aslist[3:],sameyear=False)
_,df_Nm,df_NdTm = normalize_and_average(df_HOMTN,df_NORMN,mslist)

df_GdT,df_GA,df_GdTA = normalize_and_average(df_HOMTG,df_NORMG,mlist)
_,df_Ga,df_GdTa = normalize_and_average(df_HOMTG,df_NORMG,aslist[:3],months2=aslist[3:],sameyear=False)
_,df_Gm,df_GdTm = normalize_and_average(df_HOMTG,df_NORMG,mslist)

df_JdT,df_JA,df_JdTA = normalize_and_average(df_HOMTJ,df_NORMJ,mlist)
_,df_Ja,df_JdTa = normalize_and_average(df_HOMTJ,df_NORMJ,aslist[:3],months2=aslist[3:],sameyear=False)
_,df_Jm,df_JdTm = normalize_and_average(df_HOMTJ,df_NORMJ,mslist)


## Set Font Sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dyrs = [1952,1953,1954,1995,2011,2019]

# c_cycle=['midnightblue','navy','darkblue','b','royalblue','dodgerblue','cornflowerblue','steelblue','slategray','dimgrey']
#c_cycle = ['brown','maroon','darkred','firebrick','r','tomato','salmon']
c_cycle = ['gold','goldenrod','peru','orchid','salmon','tomato']


plt.figure(figsize=(6.75,9))

### PLOT ANNUAL AVERAGE TEMPERATURE ANOMALIES
ax1 = plt.subplot(211)
for y_ in df_HOMTG['Year'].unique():
	if y_ >= 1940:
		IND = df_HOMTG['Year'] == y_
		ax1.plot(df_HOMTG[IND]['Month'],df_HOMTG[IND]['Temp(C)'],color='k',alpha=0.2)
for i_,y_ in enumerate(dyrs):
	IND = df_HOMTG['Year'] == y_
	ic = c_cycle[i_]
	
	MIND = df_HOMTG[IND]['Temp(C)'] == df_HOMTG[IND]['Temp(C)'].max()
	max_mo = df_HOMTG[IND][MIND]['Month'].values[0]
	ax1.plot(df_HOMTG[IND][MIND]['Month'],df_HOMTG[IND][MIND]['Temp(C)'],'o',color=ic,markersize=9)
	ax1.plot(df_HOMTG[IND]['Month'],df_HOMTG[IND]['Temp(C)'],label='%04d (%s)'%(y_,max_mo),color=ic)

ax1.plot(df_NORMG[df_NORMG['Yr1']==1981].T['mean'][mlist],'w:',linewidth=2,alpha=0.75)
ax1.set_ylabel('Air Temperature ($\\degree C$)')
ax1.set_ylim([-15,20])
ax1.set_xlim(['Jan','Dec'])
ax1.fill_between(['Apr','Oct'],[-15,-15],[20,20],alpha=0.1,color='maroon')
ax1.fill_between(['Jan','Apr'],[-15,-15],[20,20],alpha=0.1,color='dodgerblue')
ax1.fill_between(['Oct','Dec'],[-15,-15],[20,20],alpha=0.1,color='dodgerblue')

ax1.text('Jun',0,'Melt-Season')
ax1.text('Jan',12,'    Winter')
ax1.text('Jan',18,'  a',fontweight='extra bold',fontstyle='italic',fontsize=14)
ax1.legend(ncol=2)
ax1.set_xlabel('Month')

cyr = 30
DT_CLIM  = pd.Timedelta(cyr*365.24,unit='day')
ylims1 = [-4.5,2.5]
ax2 = plt.subplot(212)
for i_,y_ in enumerate(dyrs):
	plt.plot(np.ones(2,)*y_,list(ylims1),'.-',color=c_cycle[i_])


ax2.plot(df_GdTA['Year'],df_GdTA['Temp(C)'],color='black',label='Mean Annual',alpha=0.7)
ax2.plot(df_GdTa['Year'],df_GdTa['Temp(C)'],color='dodgerblue',label='Mean Winter')
ax2.plot(df_GdTm['Year'],df_GdTm['Temp(C)'],color='maroon',label='Mean Melt-Season')
df_GTAC = df_GdTA.copy().rolling(DT_CLIM).mean()
df_GTAC.index -= DT_CLIM/2
df_GTAC = df_GTAC[df_GTAC.index >= df_GdTA.index.min()]
ax2.plot(df_GTAC['Year'],df_GTAC['Temp(C)'],'red',label='Climate Mean (%d a)'%(cyr))
ax2.set_xlim([1940,2020])
ax2.set_ylim(ylims1)
ax2.set_ylabel('Air Temperature Anomaly\n$\\Delta T$ ($\\degree C$)')
ax2.text(1940,2,'  b',fontweight='extra bold',fontstyle='italic',fontsize=14)
ax2.plot([1940,2020],[0,0],':',color='grey',alpha=0.5)
ax2.legend(ncol=2)
ax2.set_xlabel('Year')

if issave:
	OFILE = os.path.join(ODIR,'SG7_Accepted_Fig2_Climate_and_Weather_Nordegg_%ddpi.png'%(DPI))
	plt.savefig(OFILE,dpi=DPI,format='png')   


plt.show()

