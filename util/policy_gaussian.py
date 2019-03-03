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
	

	def estimate_params(self):

		raise NotImplementedError