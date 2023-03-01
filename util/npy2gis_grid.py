import numpy as np

def write_ascii_raster(file,EE,NN,ZZ,nodata_value=-9999):
	# Open File, appending extension if needed
	if file.split('.')[-1] != 'ascii':
		fobj = open(file+'.ascii',"w")
	else:
		fobj = open(file,"w")
	# Write Header
	fobj.write('ncols %d\n'%(EE.shape[1]))
	fobj.write('nrows %d\n'%(EE.shape[0]))
	fobj.write('xllcorner %.3f\n'%(np.nanmin(EE)))
	fobj.write('yllcorner %.3f\n'%(np.nanmin(NN)))
	fobj.write('cellsize %.3f\n'%(np.nanmean(EE[:,1:] - EE[:,:-1])))
	fobj.write('nodata_value %.3f\n'%(nodata_value))

	# Iterate across input griddata
	for i_ in np.arange(EE.shape[0],0,-1)-1:
		for j_ in range(EE.shape[1]):
			z_ij = ZZ[i_,j_]
			# Check if nodata (handles NaN & Inf)
			if not np.isfinite(z_ij):
				z_ij = nodata_value
			# Write entry ij
			fobj.write('%.3f'%(z_ij))
			# If not done with the line, add a space
			if j_+1 < EE.shape[1]:
				fobj.write(' ')
			# If done with the line, but more lines are present, add a return
			elif i_ > 0:
				fobj.write('\n')
# Finish Writing
	fobj.close()


#### FETCH NPY GRIDS AND WRITE TO ASCII ####
from glob import glob

# ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/INTERPOLATION/Grids'
ROOT = '/home/nates/ActiveProjects/SGGS/MANUSCRIPT_CONTENT/INTERPOLATION/Grids'


## LOAD COMMON COORDINATES ##
# Load Easting & Northing Grid
EE = np.load(ROOT+'/Easting_Grid.npy')
NN = np.load(ROOT+'/Northing_Grid.npy')

## LOAD HVSR & HYDROPOTENTIAL MODEL DATA ##
# Load z_BED Grid
HVu = np.load(ROOT+'/HVSR_Bed_Elevation_Mean_RBF.npy')
# Load H_I Uncertainty Grid
HVo = np.load(ROOT+'/HVSR_Ice_Thickness_STD_RBF.npy')

## LOAD SURFACE VELOCITY & VERTICAL STRAIN RATE DATA ##
# # Surface Velocity Vector Field
# VSE = np.load(ROOT+'/Surface_Velocity_Easting_Mean_RBF.npy')
# VSN = np.load(ROOT+'/Surface_Velocity_Northing_Mean_RBF.npy')
# Vertical Strain Rate Field
DEZ = np.load(ROOT+'/Surface_Vertical_Strain_Mean_RBF.npy')

## LOAD MASKS ###
# Load HVSR Used Station Mask 
M_HV = np.load(ROOT+'/HVSR_Station_RBF_MASK.npy')
# Load Surface Velocity Station Mask
M_VS = np.load(ROOT+'/Surface_Velocity_RBF_MASK.npy')


## Reconstitute HVSR Layers
HV_Shallow = (HVu + 2*HVo)*M_HV
HV_Mean = HVu*M_HV
HV_Deep = (HVu - 2*HVo)*M_HV
# Write to file
write_ascii_raster(ROOT+'/HVSR_Shallow_RBF_Bed',EE,NN,HV_Shallow)
write_ascii_raster(ROOT+'/HVSR_Mean_RBF_Bed',EE,NN,HV_Mean)
write_ascii_raster(ROOT+'/HVSR_Deep_RBF_Bed',EE,NN,HV_Deep)

## Reconstitute Vertical Strain Rate Layers
DEZ_Masked = DEZ*M_VS
write_ascii_raster(ROOT+'/Surface_Vertical_Strain_Rate_RBF',EE,NN,DEZ_Masked)