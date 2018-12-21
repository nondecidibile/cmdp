import numpy as np
from util.util import onehot_encode


class BoltzmannPolicy:

	"""
	Boltzmann policy

	Parameters are in a matrix (actions) x (state_features)
	"""

	def __init__(self, nStateFeatures, nActions, paramFillValue=0.1):

		self.nActions = nActions
		self.nStateFeatures = nStateFeatures

		self.nFeatures = nActions * nStateFeatures

		self.paramsShape = (self.nActions,self.nStateFeatures)
		self.params = np.random.rand(self.paramsShape[0],self.paramsShape[1])/10 - 0.05
	

	def compute_policy(self, stateFeatures):

		assert(len(stateFeatures)==self.nStateFeatures)

		terms = np.exp(np.dot(self.params,stateFeatures))
		sum_terms = np.sum(terms)
		
		return terms/sum_terms
	

	def draw_action(self, stateFeatures):
		policy = self.compute_policy(stateFeatures)
		return np.random.choice(self.nActions, p=policy)


	def compute_log_gradient(self, stateFeatures, action):

		"""
		Compute the gradient of the log of the policy function wrt to the policy params
		"""

		assert(len(stateFeatures) == self.nStateFeatures)
		assert(action >= 0 and action < self.nActions)

		# Build state-action features
		sa_features = []
		for i in range(self.nActions):
			sa_feature = []
			for a in range(self.nActions):
				sa_feature.append(int(a == i) * np.array(stateFeatures))
			sa_features.append(np.array(sa_feature).ravel())
		sa_features = np.array(sa_features)

		terms = np.exp(np.dot(self.params,stateFeatures))

		log_gradient = sa_features[action] - np.average(sa_features, axis=0, weights=terms)
		log_gradient = log_gradient.reshape(self.paramsShape)
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
	