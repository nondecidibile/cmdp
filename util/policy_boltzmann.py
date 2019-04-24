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
		self.params = np.zeros(self.paramsShape)
	

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

		prob = np.exp(np.dot(stateFeatures, self.params.T))
		prob = prob / np.sum(prob, axis=1)[:, None]
		mean = prob[:, :, None] * stateFeatures[:, None, :]
		log_gradient = -mean
		row_index = np.arange(stateFeatures.shape[0], dtype=np.int)
		log_gradient[row_index, action] = log_gradient[row_index, action] + stateFeatures

		return log_gradient
	

	def compute_log(self, stateFeatures, action):

		terms = np.dot(stateFeatures,self.params.T) # shape=(T,nA)
		log = terms[np.arange(terms.shape[0]),action] # phi.T * theta
		terms = np.exp(terms)
		a_sum_terms = np.sum(terms,axis=1)
		log -= np.log(a_sum_terms)
		return np.sum(log)

	

	def estimate_params(self, data, optimizer, params=None, setToZero=None, epsilon=0.01, minSteps=50, maxSteps=0):

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
			
			update_size = np.linalg.norm(np.ravel(np.asarray(update_step)),2)
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

		fisherInformation = np.zeros(shape=(self.nParams,self.nParams),dtype=np.float32)

		for n,T in enumerate(eps_len):
			for t in range(T):
				sf = eps_s[n,t]
				policy = self.compute_policy(sf)

				x1 = np.zeros(shape=(self.nParams,self.nParams))
				for a in range(self.nActions):
					sa = np.zeros(shape=self.nParams)
					sa[a*self.nStateFeatures:(a+1)*self.nStateFeatures] = sf
					x1 += policy[a]*np.outer(sa,sa)

				x2_vec = np.multiply(np.tile(sf,(self.nActions,1)).T,policy).T
				x2_vec = np.ravel(x2_vec)
				x2 = np.outer(x2_vec,x2_vec)

				fisherInformation += x1-x2

		fisherInformation /= np.sum(eps_len)

		return fisherInformation