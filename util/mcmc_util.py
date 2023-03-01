import numpy as np
from scipy.stats.distributions import norm
from pyDOE import lhs

#### UNCERTAINTY PROPAGATION ROUTINES ####

def norm_lhs(mean_vector,cov_matrix, n_samps = 1000, criterion='maximin'):
	"""
	Compilation of steps from Gou's article* on MC simulations using LHS
	to provide perturbed parameter estimates given a vector of expected parameter values
	and their corresponding covariance matrix

	*TowardsDataScience article by Shuai Gou: 
	https://towardsdatascience.com/performing-uncertainty-analysis-in-three-steps-a-hands-on-guide-9110b120987e
	:: INPUTS ::
	:type mean_vector: numpy.ndarray - n, or n,1 vector
	:param mean_vector: ordered vector of mean parameter estiamtes
	:type cov_matrix: numpy.ndarray - n x n array
	:param cov_matrix: ordered covariance matrix corresponding to entries in mean_vector
	:type n_samps: int
	:para n_samps: Number of target_samples to generate
	:type criterion: str
	:param criterion: See pyDOE.lhs
	"""
	# 0) Detect the number of parameters & do sanity checks
	nfrommv = len(mean_vector)
	nfromcm,mfromcm = cov_matrix.shape
	if nfrommv == nfromcm == mfromcm:
		n_params = nfrommv
	elif nfrommv > nfromcm == nfromcm:
		print('In development, assume dropped COV matrix entries mean 0-(co)variance')
		n_params = nfrommv
	else:
		print('Incorrectly sized input distributions, reassess')
		pass
	# 1) Conduct LHS to produce Uniform PDF samples
	uni_samples = lhs(n=n_params,samples=n_samps,criterion=criterion)
	# 2) Conduct inverse transformation sampling to create standard normal distributions
	std_norm_samples = np.zeros_like(uni_samples)
	for i_ in range(n_params):
		std_norm_samples[:,i_] = norm(loc=0,scale=1).ppf(uni_samples[:,i_])
	# 3) Convert standard normals into bivariate normals
	L = np.linalg.cholesky(cov_matrix)
	target_samples = mean_vector + std_norm_samples.dot(L.T)
	return target_samples



# def rand_array(mean_array,var_array,method=np.random.normal):
# 	"""
# 	Create a randomly perturbed array using a specified random sampler
# 	"""
