import numpy as np
from util.policy import Policy


class EpsilonBoltzmannPolicy(Policy):

	"""
	Boltzmann policy

	Parameters are in a matrix (actions) x (state_features)
	"""

	def __init__(self, nStateFeatures, nActions, epsilon=0.1, paramFillValue=0.1):

		super().__init__()

		self.nActions = nActions
		self.nStateFeatures = nStateFeatures

		self.nParams = nActions * nStateFeatures

		self.epsilon = epsilon

		self.paramsShape = (self.nActions,self.nStateFeatures)
		self.params = np.zeros(self.paramsShape) #np.random.random_sample(self.paramsShape)*paramFillValue - paramFillValue/2
	

	def compute_policy(self, stateFeatures):

		assert(len(stateFeatures)==self.nStateFeatures)

		terms = np.exp(np.dot(self.params,stateFeatures))
		prob = terms / np.sum(terms)
		prob = (1 - self.epsilon) * prob + self.epsilon / self.nActions
		
		return prob
	

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

		terms = np.exp(np.dot(stateFeatures, self.params.T))
		sum_terms = np.sum(terms, axis=1)[:, None]
		terms_features = terms[:, :, None] * stateFeatures[:, None, :]

		row_index = np.arange(stateFeatures.shape[0], dtype=np.int)
		den = (1 - self.epsilon) * (terms[row_index, action] + self.epsilon / self.nActions * np.sum(terms, axis=1))

		log_gradient = -terms_features / sum_terms[:, :, None] + self.epsilon / self.nActions * terms_features / den[:, None, None]

		num = (1 - self.epsilon) * stateFeatures * terms[row_index, action][:, None]
		log_gradient[row_index, action] = log_gradient[row_index, action] + num / den[:, None]

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
		
		old_params = self.params

		flag = True
		steps = 0

		while flag:
		
			grad = np.zeros(shape=self.paramsShape, dtype=np.float32)

			for ep in data:

				for i in range(ep["a"].size):
					
					state_features = ep["s"][i]
					reward = ep["r"][i]
					action = ep["a"][i]
					
					grad += self.compute_log_gradient(state_features,action)
			
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
	