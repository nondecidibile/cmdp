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
	

	def estimate_params(self, data, optimizer, params=None, epsilon=0.01, minSteps=50, maxSteps=0):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""

		if params is not None:
			self.params = params
		else:
			self.params = np.random.rand(self.paramsShape[0],self.paramsShape[1])/10 - 0.05
		
		flag = True
		steps = 0

		while flag:
		
			grad = np.zeros(shape=self.paramsShape, dtype=np.float32)
			'''
			for ep in data:

				for i in range(ep["a"].size):
					
					state_features = ep["s"][i]
					reward = ep["r"][i]
					action = ep["a"][i]
					
					grad += self.compute_log_gradient(state_features,action)
			'''
			for ep_n,ep_len in enumerate(data["len"]):
				grad += np.sum(self.compute_log_gradient(data["s"][ep_n][0:ep_len],data["a"][ep_n][0:ep_len]))
			
			update_step = optimizer.step(grad)
			self.params = self.params + update_step
			
			update_size = np.linalg.norm(np.ravel(np.asarray(update_step)),2)
			print(steps," - Update size :",update_size)
			steps += 1
			if update_size<epsilon or steps>maxSteps:
				flag = False
			if steps<minSteps:
				flag = True

		return self.params