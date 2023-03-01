import numpy as np




def RES_RMSD(RES_VECT):
	"""
	Calculate the Root Mean Squared Deviation (RMSD) for a numpy.ndarray
	assuming the predicted series is a 0-vector matching the dimensions of
	the input RES_VECT 
	"""
	idx = np.isnan(RES_VECT)
	RMSD = np.sqrt((np.sum(RES_VECT[~idx]**2))/len(RES_VECT[~idx]))
	return RMSD


