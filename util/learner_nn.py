import numpy as np
import tensorflow as tf
from progress.bar import Bar


class nnGpomdpLearner:
	"""
	G(PO)MDP algorithm with baseline
	"""

	alg_epsilon = 1e-9

	def __init__(self, mdp, policy, gamma=0.9975):

		self.mdp = mdp
		self.gamma = gamma
		self.policy = policy

	def draw_action(self, stateFeatures):
		return self.policy.draw_action(stateFeatures)

	def optimize_gradient(self, eps, optimizer):

		nEpisodes = len(eps["len"])
		maxEpLength = max(eps["len"])

		# log gradients
		sl = np.zeros(shape=(nEpisodes, maxEpLength, self.policy.nParams), dtype=np.float32)
		dr = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)

		for n, T in enumerate(eps["len"]):

			g = np.zeros(shape=(T, self.policy.nParams))
			for t in range(T):
				g[t] = self.policy.compute_log_gradient(eps["s"][n,t],eps["a"][n,t])

			sl[n, :T] = np.cumsum(g, axis=0)
			dr[n, :T] = self.gamma ** np.arange(T) * eps["r"][n, :T]
		
		# baseline
		num = np.sum(sl * sl * dr[:, :, None], axis=0)
		den = np.sum(sl * sl, axis=0) + self.alg_epsilon
		b = num / den

		# gradients
		grads_linear = sl * (dr[:, :, None] - b[None])
		gradient_ep = np.sum(grads_linear, axis=1)

		gradient = np.mean(gradient_ep, axis=0)
		update_step = optimizer.step(gradient)
		#update_step = gradient * learningRate

		update_step = tf.split(update_step,tf.stack(self.policy.var_params))
		for i,var in enumerate(tf.trainable_variables()):
			update_step[i] = tf.reshape(update_step[i],shape=var.shape)
			self.policy.s.run(var.assign_add(update_step[i]))
		
		return gradient