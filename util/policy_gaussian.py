import numpy as np
from util.policy import Policy


class GaussianPolicy(Policy):

	"""
	Gaussian policy (with fixed diagonal covariance matrix)
	"""

	def __init__(self, nStateFeatures, actionDim, paramFillValue=0.1):

		super().__init__()

		self.nStateFeatures = nStateFeatures
		self.actionDim = actionDim

		self.nParams = self.actionDim*self.nStateFeatures

		self.paramsShape = (self.actionDim, self.nStateFeatures)
		self.params = np.zeros(self.paramsShape) #np.random.random_sample(self.paramsShape)*paramFillValue - paramFillValue/2

		self.covarianceMatrix = 0.25 * np.eye(self.actionDim) #np.diag(np.full(self.actionDim,0.25,dtype=np.float32))
		self.cov_diag = np.diag(self.covarianceMatrix)

	def draw_action(self, stateFeatures):

		assert(len(stateFeatures)==self.nStateFeatures)

		mean = np.dot(self.params,stateFeatures)
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
	

	def estimate_params(self, data, optimizer, params=None, setToZero=None, epsilon=0.01, minSteps=50, maxSteps=0, printInfo=True):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""

		if params is not None:
			self.params = params
		else:
			self.params = np.zeros(shape=self.paramsShape)
		
		if setToZero is not None:
			self.params[:,setToZero] = 0
		
		flag = True
		steps = 0

		while flag:
		
			grad = np.zeros(shape=self.paramsShape, dtype=np.float32)

			for ep_n,ep_len in enumerate(data["len"]):
				grad += np.sum(self.compute_log_gradient(data["s"][ep_n][0:ep_len],data["a"][ep_n][0:ep_len]),axis=0)
			
			if setToZero is not None:
				grad[:,setToZero] = 0

			update_step = optimizer.step(grad)
			self.params = self.params + update_step
			update_size = np.abs(np.max(np.ravel(np.asarray(update_step))))
			if printInfo:
				print(steps," - Update size :",update_size)
			steps += 1
			if update_size<epsilon or steps>maxSteps:
				flag = False
			if steps<minSteps:
				flag = True
		
		if setToZero is not None:
			self.params[:,setToZero] = 0
		
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

			log_likelihood += self.compute_log(sf_episode,a_episode)
		
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