import numpy as np
from util.policy import Policy
import tensorflow as tf


class NeuralNetworkPolicy(Policy):

	"""
	Gaussian Policy with mean defined by a neural network
	"""

	def __init__(self, nStateFeatures, actionDim, nHiddenNeurons=8, paramInitMaxVal=0.1, gamma=0.9975):

		super().__init__()
		self.nStateFeatures = nStateFeatures
		self.actionDim = actionDim
		self.nHiddenNeurons = nHiddenNeurons
		self.gamma = gamma
		self.covariance = 0.25
		self.covarianceMatrix = self.covariance * np.eye(self.actionDim)

		tf.logging.set_verbosity(tf.logging.ERROR)

		self.inputLayer = tf.placeholder(shape=[nStateFeatures],dtype=tf.float32,name="input_ph")

		initializer = tf.initializers.random_uniform(-paramInitMaxVal, paramInitMaxVal)
		bias = tf.constant(1.0,dtype=tf.float32,shape=[1])
		self.w1 = tf.get_variable("w1", [nStateFeatures+1,nHiddenNeurons], dtype=tf.float32, initializer=initializer)
		self.w2 = tf.get_variable("w2", [nHiddenNeurons+1,actionDim], dtype=tf.float32, initializer=initializer)

		hiddenLayer = tf.tanh(tf.matmul(tf.concat([self.inputLayer[None,:],bias[None,:]],1),self.w1))
		self.outputLayer = tf.matmul(tf.concat([hiddenLayer,bias[None,:]],1),self.w2)

		self.nParams = 0
		self.var_params = []
		for var in tf.trainable_variables():
			self.var_params.append(np.prod(var.shape))
			self.nParams += np.prod(var.shape)

		self.action_ph = tf.placeholder(shape=[actionDim],dtype=tf.float32,name="action_ph")

		ratio = 1.0/self.covariance*(self.action_ph-self.outputLayer)
		self.log_gradient_w1 = tf.gradients(self.outputLayer, self.w1, grad_ys=ratio)
		self.log_gradient_w2 = tf.gradients(self.outputLayer, self.w2, grad_ys=ratio)

		init_op = tf.global_variables_initializer()
		self.s = tf.Session()
		self.s.run(init_op)
	
	def draw_action(self, stateFeatures):
		sf = np.reshape(stateFeatures,newshape=(-1))
		output = self.s.run(self.outputLayer,feed_dict={self.inputLayer:sf})
		mean = np.reshape(np.array(output),newshape=(-1))
		action = np.random.multivariate_normal(mean,self.covarianceMatrix)
		return action
	
	def compute_log_gradient(self, stateFeatures, actions):
		log_grad_w1, log_grad_w2 = self.s.run([self.log_gradient_w1,self.log_gradient_w2],feed_dict={
			self.inputLayer: stateFeatures,
			self.action_ph: actions
		})
		log_grad_w1 = np.reshape(log_grad_w1,newshape=(-1))
		log_grad_w2 = np.reshape(log_grad_w2,newshape=(-1))
		return np.concatenate([log_grad_w1,log_grad_w2])

	def optimize_gradient(self, eps, learningRate):

		nEpisodes = len(eps["len"])
		maxEpLength = max(eps["len"])

		# log gradients
		sl = np.zeros(shape=(nEpisodes, maxEpLength, self.nParams), dtype=np.float32)
		dr = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)

		for n, T in enumerate(eps["len"]):

			g = np.zeros(shape=(T, self.nParams))
			for t in range(T):
				g[t] = self.compute_log_gradient(eps["s"][n,t],eps["a"][n,t])

			sl[n, :T] = np.cumsum(g, axis=0)
			dr[n, :T] = self.gamma ** np.arange(T) * eps["r"][n, :T]
		
		# baseline
		num = np.sum(sl * sl * dr[:, :, None], axis=0)
		den = np.sum(sl * sl, axis=0) + 1e-9
		b = num / den

		# gradients
		grads_linear = sl * (dr[:, :, None] - b[None])
		gradient_ep = np.sum(grads_linear, axis=1)

		gradient = np.mean(gradient_ep, axis=0)

		var_gradient = tf.split(gradient,tf.stack(self.var_params))
		for i,var in enumerate(tf.trainable_variables()):
			var_gradient[i] = tf.reshape(var_gradient[i],shape=var.shape)
			self.s.run(var.assign_add(var_gradient[i]*learningRate))
		
		return gradient
	
	def print_params(self):
		print(self.s.run([self.w1,self.w2]))