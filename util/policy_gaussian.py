import numpy as np
from util.policy import Policy


class GaussianPolicy(Policy):

	"""
	Gaussian policy (with fixed diagonal covariance matrix)
	"""

	def __init__(self, nStateFeatures, actionDim):

		super().__init__()

		self.nStateFeatures = nStateFeatures
		self.actionDim = actionDim

		self.nParams = self.actionDim*self.nStateFeatures

		self.paramsShape = (self.actionDim, self.nStateFeatures)
		self.params = np.zeros(self.paramsShape)

		self.covarianceMatrix = 0.02 ** 2 * np.eye(self.actionDim)
		self.cov_diag = np.diag(self.covarianceMatrix)
	
	def init_random_params(self, stddev=1.0):
		self.params = stddev*np.random.rand(self.actionDim, self.nStateFeatures)

	def draw_action(self, stateFeatures):

		#print(stateFeatures)

		assert(len(stateFeatures)==self.nStateFeatures)
		mean = np.dot(self.params, stateFeatures)
		#print(stateFeatures, mean)

		action = np.random.multivariate_normal(mean,self.covarianceMatrix)
		return action


	def compute_log_gradient(self, stateFeatures, action):

		"""
		Compute the gradient of the log of the policy function wrt to the policy params
		"""
		'''
		assert(len(stateFeatures) == self.nStateFeatures)
		assert(len(action) == self.actionDim)

		mean = np.dot(self.params,stateFeatures)
		log_gradient = np.tile(stateFeatures,(self.actionDim,1))
		aa = (action-mean)/np.square(np.diag(self.covarianceMatrix))
		for i,a in enumerate(aa):
			log_gradient[i]*=a
		return log_gradient
		'''

		if np.ndim(action) == 1:
			action = action[:, None]

		mean = np.dot(stateFeatures, self.params.T)
		ratio = (action - mean) / self.cov_diag[None, :]
		log_gradient = ratio[:, :, None] * stateFeatures[:, None, :]
		return log_gradient
	

	def compute_log(self, stateFeatures, action):

		covinv = np.linalg.inv(self.covarianceMatrix)
		covdet = np.linalg.det(self.covarianceMatrix)

		mean = np.dot(stateFeatures, self.params.T) # shape=(T,nA)
		v = action-mean
		terms = -0.5*np.einsum('ij,ij->i', np.dot(v, covinv), v)
		sumterms = np.sum(terms)
		return sumterms - action.shape[0]*0.5*np.log(4*(np.pi**2)*covdet)
	

	def estimate_params(self, data, setToZero=None):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""
		
		n = np.sum(data["len"])

		X = np.zeros(shape=(n,self.nStateFeatures),dtype=np.float32)
		A = np.zeros(shape=(n,self.actionDim),dtype=np.float32)

		i = 0
		for ep_n,ep_len in enumerate(data["len"]):
			X[i:i+ep_len] = data["s"][ep_n][0:ep_len]
			A[i:i+ep_len] = data["a"][ep_n][0:ep_len]
			i += ep_len

		if setToZero is not None:
			X = np.delete(X,setToZero,1)
		
		self.params = np.linalg.lstsq(X, A, rcond=None)[0].T
		#print(self.params)
		#print(np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),A).T)

		if setToZero is not None:
			self.params = np.insert(self.params,setToZero,0,1)

		return self.params
	

	def getLogLikelihood(self, data, params=None):

		if params is not None:
			self.params = params
		
		eps_s = data["s"]
		eps_a = data["a"]
		eps_len = data["len"]

		log_likelihood = 0

		for n,T in enumerate(eps_len):
			sf_episode = eps_s[n]
			a_episode = eps_a[n]

			log_likelihood += self.compute_log(sf_episode[0:T],a_episode[0:T])
		
		return log_likelihood


	def getAnalyticalFisherInformation(self, data):
		
		eps_s = data["s"]
		eps_len = data["len"]

		fisherInformation = np.zeros(shape=(self.nStateFeatures,self.nStateFeatures),dtype=np.float32)

		for n,T in enumerate(eps_len):
			sf = eps_s[n,:T]
			f = np.matmul(sf.T,sf)
			fisherInformation += f

		fisherInformation /= np.sum(eps_len) * 0.25

		return fisherInformation