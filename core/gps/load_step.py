import pandas as pd
import os


def load(stage_int,vers=None):
	DROOT = os.path.join('..','..','..','..')
	if stage_int == 0 and vers is None:
		print('Loading Post Processing Outputs: Stage 0')
		IDATA = os.path.join(DROOT,'data','MINDSatUW','GPS','POST_PROCESSED','continuous','rov1_cat.pos')
	elif stage_int == 1 and vers == 'A':
		print('Loading Localized, Unstitched Data Outputs: Stage 1A')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1A_rov1_localized.csv')
	elif stage_int == 1 and vers == 'B':
		print('Loading Localized, Stitched Data Outputs: Stage 1B')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1B_rov1_stitched.csv')
	elif stage_int == 1 and vers == 'C':
		print('Loading Localized, Stitched Data Outputs: Stage 1C')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1C_rov1_downsampled.csv')
	elif stage_int == 2 and vers is None:
		print('Loading Iteratively Despiked Data Outputs: Stage 2')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S2_rov1_despiked.csv')
	elif stage_int == 2 and vers == 'C':
		print('Loading Iteratively Despiked Data Outputs: Stage 2C')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S2C_rov1_despiked.csv')
	elif stage_int == 3 and vers is None:
		print('Loading Rotated Data Outputs: Stage 3')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S3_rov1_rotated.csv')
	elif stage_int == 3 and vers == 'C':
		print('Loading Rotated Data Outputs: Stage 3C')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S3C_rov1_rotated.csv')
	elif stage_int == 4 and vers is None:
		print('Loading Velocity Estimates: Stage 4')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S4_rov1_vels.csv')
	elif stage_int == 4 and vers == 'C':
		print('Loading Velocity Estimates: Stage 4')
		IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S4C_rov1_vels.csv')
	# Actually Load Data
	try:
		df = pd.read_csv(IDATA,parse_dates=True,index_col='GPST')
	except:
		df = pd.read_csv(IDATA,parse_dates=True,index_col=0)
	return df