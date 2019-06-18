import numpy as np
from util.policy import Policy
import tensorflow as tf
from random import sample 


class nnGaussianPolicy(Policy):

	"""
	Gaussian Policy with mean defined by a neural network
	"""

	def __init__(self, nStateFeatures, actionDim, nHiddenNeurons=8, paramInitMaxVal=0.01, variance=0.25):
		super().__init__()
		self.nStateFeatures = nStateFeatures
		self.actionDim = actionDim
		self.nHiddenNeurons = nHiddenNeurons
		self.covariance = np.float32(variance)
		self.covarianceMatrix = self.covariance * np.eye(self.actionDim,dtype=np.float32)

		tf.logging.set_verbosity(tf.logging.ERROR)

		# Placeholders
		self.inputLayer = tf.placeholder(shape=[None,nStateFeatures],dtype=tf.float32,name="input_ph")
		self.action_ph = tf.placeholder(shape=[None,actionDim],dtype=tf.float32,name="action_ph")

		# Params
		initializer = tf.initializers.random_uniform(-paramInitMaxVal, paramInitMaxVal)
		self.params = {
			"w1": tf.get_variable("w1", [nStateFeatures,nHiddenNeurons], dtype=tf.float32, initializer=initializer),
			"b1": tf.get_variable("b1", [nHiddenNeurons], dtype=tf.float32, initializer=initializer),
			"w2": tf.get_variable("w2", [nHiddenNeurons,actionDim], dtype=tf.float32, initializer=initializer),
			"b2": tf.get_variable("b2", [actionDim], dtype=tf.float32, initializer=initializer)
		}
		self.nParams = 0
		self.var_params = []
		for var in tf.trainable_variables():
			self.var_params.append(np.prod(var.shape))
			self.nParams += np.prod(var.shape)

		# Network structure
		hiddenLayer = tf.tanh(tf.add(tf.matmul(self.inputLayer,self.params["w1"]),self.params["b1"]))
		self.outputLayer = tf.add(tf.matmul(hiddenLayer,self.params["w2"]),self.params["b2"])

		# Gradient (Log Policy)
		ratio = 1.0/self.covariance*(self.action_ph-self.outputLayer)
		self.log_gradient = {
			"w1": tf.gradients(self.outputLayer, self.params["w1"], grad_ys=ratio),
			"b1": tf.gradients(self.outputLayer, self.params["b1"], grad_ys=ratio),
			"w2": tf.gradients(self.outputLayer, self.params["w2"], grad_ys=ratio),
			"b2": tf.gradients(self.outputLayer, self.params["b2"], grad_ys=ratio)
		}

		# Log Policy
		covinv = np.linalg.inv(self.covarianceMatrix)
		covdet = np.linalg.det(self.covarianceMatrix)
		v = self.action_ph-self.outputLayer
		terms = -0.5*tf.einsum('ij,ij->i', tf.matmul(v, covinv), v)
		sumterms = tf.reduce_sum(terms)
		self.log_policy = sumterms-tf.multiply(tf.cast(tf.shape(self.action_ph)[0],tf.float32),0.5*np.log(4*(np.pi**2)*covdet))
		# Log Likelihood
		self.neg_log_likelihood = -1.0 * self.log_policy

		# Init operation
		self.init_op = tf.global_variables_initializer()
		self.s = tf.Session()
		self.s.run(self.init_op)
	
	def draw_action(self, stateFeatures):
		sf = np.reshape(stateFeatures,[-1,self.nStateFeatures])
		output = self.s.run(self.outputLayer,feed_dict={self.inputLayer:sf})
		mean = np.reshape(np.array(output),newshape=(-1))
		action = np.random.multivariate_normal(mean,self.covarianceMatrix)
		return action
	
	def compute_log_gradient(self, stateFeatures, actions):
		log_grad = self.s.run(self.log_gradient, feed_dict={
			self.inputLayer: np.reshape(stateFeatures,[-1,self.nStateFeatures]),
			self.action_ph: np.reshape(actions,[-1,self.actionDim])
		})
		log_grad_list = [np.reshape(v,newshape=(-1)) for k,v in log_grad.items()] 
		return np.concatenate(log_grad_list)

	def estimate_params(self, data, lr, params0=None, nullFeature=None, batchSize=25, epsilon=0.01, minSteps=50, maxSteps=0, printInfo=True):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""
		N = range(len(data["len"]))
		optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999)
		train_op = optimizer.minimize(loss=self.neg_log_likelihood)

		old_params = self.get_params()
		if nullFeature is not None:
			w = self.params["w1"].eval(self.s)
			w[nullFeature,:] = 0
			self.s.run(self.params["w1"].assign(w))
		
		init_op = tf.global_variables_initializer()
		self.s.run(init_op)

		if params0 is not None:
			self.set_params(params0)

		flag = True
		steps = 0

		while flag:

			#grad = np.zeros(shape=self.nParams, dtype=np.float32)
			batch_indexes = sample(N,batchSize)
			data_s = np.concatenate([data["s"][ep_n][:data["len"][ep_n]] for ep_n in batch_indexes])
			data_a = np.concatenate([data["a"][ep_n][:data["len"][ep_n]] for ep_n in batch_indexes])

			self.s.run(train_op,feed_dict={
				self.inputLayer: np.reshape(data_s,newshape=(-1,self.nStateFeatures)),
				self.action_ph: np.reshape(data_a,newshape=(-1,self.actionDim))
			})

			if nullFeature is not None:
				w = self.params["w1"].eval(self.s)
				w[nullFeature,:] = 0
				self.s.run(self.params["w1"].assign(w))
			params = self.get_params()
			update_size = np.linalg.norm(np.ravel(np.asarray(params-old_params)),np.inf)
			old_params = params

			if printInfo:
				print(steps," - Update size :",update_size)
			steps += 1
			if update_size<epsilon or steps>maxSteps:
				flag = False
			if steps<minSteps:
				flag = True
		
		return params

	def compute_log(self, stateFeatures, action):
		return self.s.run(self.log_policy,feed_dict={
			self.inputLayer: np.reshape(stateFeatures,[-1,self.nStateFeatures]),
			self.action_ph: np.reshape(action,[-1,self.actionDim])
		})

	def getLogLikelihood(self, data):
		eps_s = data["s"]
		eps_a = data["a"]
		eps_len = data["len"]

		log_likelihood = 0

		for n,T in enumerate(eps_len):
			log_likelihood += np.sum(self.compute_log(eps_s[n,:T],eps_a[n,:T]))
		
		return log_likelihood
	
	def set_params(self, params):
		new_params = tf.split(params,tf.stack(self.var_params))
		for i,var in enumerate(tf.trainable_variables()):
			new_params[i] = tf.reshape(new_params[i],shape=var.shape)
			self.s.run(var.assign(new_params[i]))
	
	def get_params(self):
		params = self.s.run(self.params)
		params_list = [np.reshape(v,newshape=(-1)) for k,v in params.items()] 
		return np.concatenate(params_list)
	
	def print_params(self):
		print(self.s.run(self.params))