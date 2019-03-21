import numpy as np
from util.policy import Policy


class BoltzmannPolicy(Policy):

	"""
	Boltzmann policy

	Parameters are in a matrix (actions) x (state_features)
	"""

	def __init__(self, nStateFeatures, nActions, paramFillValue=0.1):

		super().__init__()

		self.nActions = nActions
		self.nStateFeatures = nStateFeatures

		self.nParams = nActions * nStateFeatures

		self.paramsShape = (self.nActions,self.nStateFeatures)
		self.params = np.zeros(self.paramsShape) #np.random.random_sample(self.paramsShape)*paramFillValue - paramFillValue/2
	

	def compute_policy(self, stateFeatures):

		assert(len(stateFeatures)==self.nStateFeatures)

		terms = np.exp(np.dot(self.params,stateFeatures))
		sum_terms = np.sum(terms)
		
		return terms/sum_terms
	

	def draw_action(self, stateFeatures):
		policy = self.compute_policy(stateFeatures)
		#print(policy.max())
		return np.random.choice(self.nActions, p=policy)


	def compute_log_gradient(self, stateFeatures, action):

		"""
		Compute the gradient of the log of the policy function wrt to the policy params
		"""

		'''
		assert(len(stateFeatures) == self.nStateFeatures)
		assert(action >= 0 and action < self.nActions)
		
		# Build state-action features
		sa_features = np.zeros((self.nActions, self.nParams))
		row_index = np.repeat(np.arange(self.nActions, dtype=np.int), self.nStateFeatures)
		col_index = np.arange(self.nParams, dtype=np.int)
		sa_features[row_index, col_index] = np.tile(stateFeatures, self.nActions)

		sa_features_old = []
		for i in range(self.nActions):
			sa_feature = []
			for a in range(self.nActions):
				sa_feature.append(int(a == i) * np.array(stateFeatures))
			sa_features_old.append(np.array(sa_feature).ravel())
		sa_features_old = np.array(sa_features_old)

		assert np.allclose(sa_features, sa_features_old)


		prob = np.exp(np.dot(self.params, stateFeatures))
		prob = prob / np.sum(prob)

		mean = np.outer(prob, stateFeatures)
		log_gradient = -mean
		log_gradient[action] = log_gradient[action] + stateFeatures


		terms = np.exp(np.dot(self.params,stateFeatures))

		log_gradient_old = sa_features[action] - np.average(sa_features, axis=0, weights=terms)
		log_gradient_old = log_gradient_old.reshape(self.paramsShape)

		assert np.allclose(log_gradient, log_gradient_old)
		'''

		'''
		For one (s,a) at a time
		prob = np.exp(np.dot(self.params, stateFeatures))
		prob = prob / np.sum(prob)
		mean = np.outer(prob, stateFeatures)
		log_gradient = -mean
		log_gradient[action] = log_gradient[action] + stateFeatures
		'''

		prob = np.exp(np.dot(stateFeatures, self.params.T))
		prob = prob / np.sum(prob, axis=1)[:, None]
		mean = prob[:, :, None] * stateFeatures[:, None, :]
		log_gradient = -mean
		row_index = np.arange(stateFeatures.shape[0], dtype=np.int)
		log_gradient[row_index, action] = log_gradient[row_index, action] + stateFeatures

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
			self.params = np.zeros(shape=self.paramsShape)
		
		old_params = self.params

		flag = True
		steps = 0

		while flag:
		
			grad = np.zeros(shape=self.paramsShape, dtype=np.float32)

			for ep_n,ep_len in enumerate(data["len"]):
				grad += np.sum(self.compute_log_gradient(data["s"][ep_n][0:ep_len],data["a"][ep_n][0:ep_len]),axis=0)
			
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
	