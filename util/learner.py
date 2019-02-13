import numpy as np
from util.policy import BoltzmannPolicy
from util.util import build_gridworld_features

class GpomdpLearner:

	"""
	G(PO)MDP algorithm with baseline
	"""

	def __init__(self, mdp, nStateFeatures, nActions, gamma=0.99):

		self.nStateFeatures = nStateFeatures
		self.nActions = nActions

		self.gamma = gamma

		self.policy = BoltzmannPolicy(nStateFeatures,nActions)


	def draw_action(self, stateFeatures):
		return self.policy.draw_action(stateFeatures)


	def estimate_gradient(self, data, getSampleVariance=False):

		"""
		Compute the gradient of J wrt to the policy params
		"""

		# Compute all the log-gradients
		nEpisodes = len(data)
		epLength = [ep["a"].size for ep in data]
		maxEpLength = max(epLength)
		logGradients = np.zeros(shape=(nEpisodes,maxEpLength,self.policy.nFeatures))
		for n,ep in enumerate(data):
			for i in range(ep["a"].size):
				g = self.policy.compute_log_gradient(ep["s"][i],ep["a"][i])
				logGradients[n,i] = np.ravel(np.asarray(g))

		#
		# Compute the baseline
		#
		
		baseline = np.zeros(shape=(maxEpLength,self.policy.nFeatures))
		
		for j in range(maxEpLength):

			episodes = (np.asarray(data))
			
			num = np.zeros(shape=self.policy.nFeatures, dtype=np.float32)
			den = np.zeros(shape=self.policy.nFeatures, dtype=np.float32)

			for n,ep in enumerate(episodes):
				
				log_g = np.sum(logGradients[n,0:j+1],axis=0)
				square_log_g = log_g ** 2

				num += square_log_g * (np.power(self.gamma,j)*ep["r"][j] if len(ep["r"])>j else 0)
				den += square_log_g

			baseline[j] = np.divide(num,den+1e-09)
		
		#
		# Compute the gradient
		#

		gradient = np.zeros(shape=self.policy.paramsShape, dtype=np.float32)
		grads = np.zeros(shape=np.concatenate([[nEpisodes],self.policy.paramsShape]), dtype=np.float32)

		for n,ep in enumerate(data):

			sum_log_grad = np.zeros(shape=self.policy.paramsShape, dtype=np.float32)

			for i in range(ep["a"].size):
				
				state_features = ep["s"][i]
				reward = ep["r"][i]
				action = ep["a"][i]
				
				log_grad = np.reshape(logGradients[n,i], newshape=self.policy.paramsShape)
				sum_log_grad = sum_log_grad + log_grad
				
				baseln = np.reshape(baseline[i],newshape=self.policy.paramsShape)
				grads[n] = grads[n] + sum_log_grad * (np.power(self.gamma,i)*reward - baseln)
			
			gradient = gradient + grads[n]
		
		gradient = gradient/nEpisodes
		if not getSampleVariance:
			return gradient
		
		#
		# Compute the sample variance
		#

		variance = np.zeros(shape=self.policy.paramsShape, dtype=np.float32)
		for i in range(nEpisodes):
			variance += np.square(grads[i]-gradient)
		variance = variance/nEpisodes

		return (gradient,variance)
	