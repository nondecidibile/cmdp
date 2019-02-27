import numpy as np
from util.util import build_gridworld_features
from progress.bar import Bar

class GpomdpLearner:

	"""
	G(PO)MDP algorithm with baseline
	"""

	def __init__(self, mdp, policy, gamma=0.99):

		self.mdp = mdp
		self.gamma = gamma

		self.policy = policy


	def draw_action(self, stateFeatures):
		return self.policy.draw_action(stateFeatures)


	def estimate_gradient(self, data, getSampleVariance=False, showProgress=False):

		"""
		Compute the gradient of J wrt to the policy params
		"""

		eps_s = data["s"]
		eps_a = data["a"]
		eps_r = data["r"]
		eps_len = data["len"]

		# Compute all the log-gradients
		nEpisodes = len(eps_len)
		maxEpLength = max(eps_len)

		if showProgress:
			bar = Bar('Computing gradient', max=nEpisodes)
		
		sl = np.zeros(shape=(nEpisodes,maxEpLength,self.policy.nParams),dtype=np.float32)
		dr = np.zeros(shape=(nEpisodes,maxEpLength,self.policy.nParams),dtype=np.float32)

		for n,T in enumerate(eps_len):
			for i in range(T):
				g = np.ravel(self.policy.compute_log_gradient(eps_s[n,i],eps_a[n,i]))				
				sl[n,i] = (g if i==0 else sl[n,i-1]+g)
				dr[n,i] = np.full(shape=self.policy.nParams,fill_value=(self.gamma**i)*eps_r[n,i])
			for j in range(T,maxEpLength):
				sl[n,j] = sl[n,j-1]
			if showProgress:
				bar.next()
		
		if showProgress:
			bar.finish()

		#
		# Compute the baseline
		#
		
		num = np.sum(sl*sl*dr,axis=0)
		den = np.sum(sl*sl,axis=0)+1e-09
		b = num/den


		#
		# Compute the gradient
		#

		grads_linear = sl*(dr-b)
		gradient_ep_linear = np.sum(grads_linear,axis=1)/nEpisodes
		gradient_linear = np.sum(gradient_ep_linear,axis=0)
		gradient = np.reshape(gradient_linear,newshape=self.policy.paramsShape)
		
		if not getSampleVariance:
			return gradient
		
		#
		# Compute the sample variance
		#

		variance = np.zeros(shape=self.policy.paramsShape, dtype=np.float32)
		for i in range(nEpisodes):
			variance += np.reshape(np.square(gradient_ep_linear[i]-gradient_linear),newshape=self.policy.paramsShape)
		variance = variance/nEpisodes

		return (gradient,variance)
	