import numpy as np
from util.policy import Policy
import tensorflow as tf
from random import sample 
from scipy.stats import multivariate_normal


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

	def estimate_params(self, data, lr=0.03, params0=None, nullFeatures=None, lockExcept=None, batchSize=100, epsilon=1e-6, minSteps=50, maxSteps=10000, printInfo=True):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""
		N = len(data["len"])
		Nrange = range(N)
		optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999)
		train_op = optimizer.minimize(loss=self.neg_log_likelihood)

		if nullFeatures is not None:
			setToZeroOp = tf.scatter_update(self.params["w1"],nullFeatures,0)
		
		init_op = tf.global_variables_initializer()
		self.s.run(init_op)

		if params0 is not None:
			self.set_params(params0)

		if nullFeatures is not None:
			self.s.run(setToZeroOp)
		
		if lockExcept is not None:
			lockMask = np.ones(shape=self.nStateFeatures,dtype=np.bool)
			lockedIndices = np.arange(self.nStateFeatures)
			lockMask[lockExcept] = False
			lockedParams = self.s.run(self.params["w1"])
			resetLockOp = tf.scatter_update(self.params["w1"],lockedIndices[lockMask],lockedParams[lockMask])
			train_op = optimizer.minimize(loss=self.neg_log_likelihood,var_list=[self.params["w1"]])

		flag = True
		steps = 0
		bestParams = np.copy(self.get_params())
		bestLL = -np.inf
		avg_update_size = 1

		while flag:

			#grad = np.zeros(shape=self.nParams, dtype=np.float32)
			batch_indexes = sample(Nrange,np.min([batchSize,N]))
			data_s = np.concatenate([data["s"][ep_n][:data["len"][ep_n]] for ep_n in batch_indexes])
			data_a = np.concatenate([data["a"][ep_n][:data["len"][ep_n]] for ep_n in batch_indexes])

			self.s.run(train_op,feed_dict={
				self.inputLayer: np.reshape(data_s,newshape=(-1,self.nStateFeatures)),
				self.action_ph: np.reshape(data_a,newshape=(-1,self.actionDim))
			})
			
			if nullFeatures is not None:
				self.s.run(setToZeroOp)
			
			if lockExcept is not None:
				self.s.run(resetLockOp)
			
			if steps%25==0:
				ll = self.getLogLikelihood(data)
				avg_update_size = 0.1*(ll-bestLL)+0.9*avg_update_size if (ll>bestLL and bestLL>-np.inf) else 0.9*avg_update_size
				if ll>bestLL:
					bestLL = ll
					bestParams = np.copy(self.get_params())

				if printInfo:
					print(steps,"- ll = ",ll)
					print("avg_ll_update_size = ",avg_update_size)
			
			steps += 1
			if avg_update_size<epsilon or steps>maxSteps:
				flag = False
			if steps<minSteps:
				flag = True
		
		return bestParams

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
	
	def getKlDivergence(self, data, paramsP, paramsQ):
		eps_s = data["s"]
		eps_a = data["a"]
		eps_len = data["len"]

		N = eps_len.size
		Tmax = np.max(eps_len)

		px = np.zeros(shape=(N,Tmax),dtype=np.float32)
		logpx = np.zeros(shape=(N,Tmax),dtype=np.float32)
		logqx = np.zeros(shape=(N,Tmax),dtype=np.float32)

		self.set_params(paramsP)
		for n,T in enumerate(eps_len):
			logpx[n] = self.compute_log(eps_s[n,:T],eps_a[n,:T])
			sf = np.reshape(eps_s[n],[-1,self.nStateFeatures])
			a = np.reshape(eps_a[n],[-1,self.actionDim])
			output = self.s.run(self.outputLayer,feed_dict={self.inputLayer:sf})
			#for t in range(eps_len[n]):
			#	px[n,t] = multivariate_normal.pdf(a[t], mean=output[t], cov=self.covariance)
				
		self.set_params(paramsQ)
		for n,T in enumerate(eps_len):
			logqx[n] = self.compute_log(eps_s[n,:T],eps_a[n,:T])
		
		return np.sum(logpx-logqx)
	
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