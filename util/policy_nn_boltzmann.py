import numpy as np
from util.policy import Policy
import tensorflow as tf


class nnBoltzmannPolicy(Policy):

	"""
	Boltzmann Policy with mean defined by a neural network
	"""

	def __init__(self, nStateFeatures, nActions, nHiddenNeurons=8, paramInitMaxVal=0.01):
		super().__init__()
		self.nStateFeatures = nStateFeatures
		self.nActions = nActions
		self.nHiddenNeurons = nHiddenNeurons

		tf.logging.set_verbosity(tf.logging.ERROR)

		self.inputLayer = tf.placeholder(shape=[nStateFeatures],dtype=tf.float32,name="input_ph")

		initializer = tf.initializers.random_uniform(-paramInitMaxVal, paramInitMaxVal)
		bias = tf.constant(1.0,dtype=tf.float32,shape=[1])
		self.w1 = tf.get_variable("w1", [nStateFeatures+1,nHiddenNeurons], dtype=tf.float32, initializer=initializer)
		self.w2 = tf.get_variable("w2", [nHiddenNeurons+1,nActions], dtype=tf.float32, initializer=initializer)

		hiddenLayer = tf.tanh(tf.matmul(tf.concat([self.inputLayer[None,:],bias[None,:]],1),self.w1))
		self.outputLayer = tf.nn.softmax(tf.matmul(tf.concat([hiddenLayer,bias[None,:]],1),self.w2))

		self.nParams = 0
		self.var_params = []
		for var in tf.trainable_variables():
			self.var_params.append(np.prod(var.shape))
			self.nParams += np.prod(var.shape)

		self.action_ph = tf.placeholder(shape=[nActions],dtype=tf.float32,name="action_ph")

		#ratio = 1.0/self.covariance*(self.action_ph-self.outputLayer)
		self.log_gradient_w1 = tf.gradients(self.outputLayer, self.w1, grad_ys=self.action_ph)
		self.log_gradient_w2 = tf.gradients(self.outputLayer, self.w2, grad_ys=self.action_ph)

		init_op = tf.global_variables_initializer()
		self.s = tf.Session()
		self.s.run(init_op)
	
	def draw_action(self, stateFeatures):
		sf = np.reshape(stateFeatures,newshape=(-1))
		output = self.s.run(self.outputLayer,feed_dict={self.inputLayer:sf})
		return np.random.choice(self.nActions, p=np.ravel(output))
	
	def compute_log_gradient(self, stateFeatures, actions):
		log_grad_w1, log_grad_w2 = self.s.run([self.log_gradient_w1,self.log_gradient_w2],feed_dict={
			self.inputLayer: stateFeatures,
			self.action_ph: actions
		})
		log_grad_w1 = np.reshape(log_grad_w1,newshape=(-1))
		log_grad_w2 = np.reshape(log_grad_w2,newshape=(-1))
		return np.concatenate([log_grad_w1,log_grad_w2])
	
	def print_params(self):
		print(self.s.run([self.w1,self.w2]))