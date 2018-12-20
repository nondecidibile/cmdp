import numpy as np


class AdamOptimizer:

	"""
	ADAM optimizer
	"""

	def __init__(self, shape, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):

		self.lr = learning_rate

		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon

		self.m_t = np.zeros(shape=shape, dtype=np.float32)
		self.v_t = np.zeros(shape=shape, dtype=np.float32)

		self.beta1_t = self.beta1
		self.beta2_t = self.beta2


	def step(self, gradient):

		self.m_t = self.beta1*self.m_t + (1.0-self.beta1)*gradient
		self.v_t = self.beta2*self.v_t + (1.0-self.beta2)*(gradient**2)

		self.beta1_t *= self.beta1
		self.beta2_t *= self.beta2

		m_t_hat = self.m_t/(1.0-self.beta1_t)
		v_t_hat = self.v_t/(1.0-self.beta2_t)

		step = self.lr * m_t_hat / (v_t_hat**0.5 + self.epsilon)
		return step

