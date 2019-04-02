import numpy as np
from progress.bar import Bar


class GpomdpLearner:
    """
	G(PO)MDP algorithm with baseline
	"""

    alg_epsilon = 1e-9

    def __init__(self, mdp, policy, gamma=0.99):

        self.mdp = mdp
        self.gamma = gamma

        self.policy = policy

    def draw_action(self, stateFeatures):
        return self.policy.draw_action(stateFeatures)

    def estimate_gradient(self, data, initialIS=None, getSampleVariance=False, showProgress=False, getEstimates=False):

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

        sl = np.zeros(shape=(nEpisodes, maxEpLength, self.policy.nParams), dtype=np.float32)
        dr = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)

        '''
		dr = np.zeros(shape=(nEpisodes, maxEpLength, self.policy.nParams), dtype=np.float32)
		for n, T in enumerate(eps_len):
			for i in range(T):
				g = np.ravel(self.policy.compute_log_gradient(eps_s[n,i],eps_a[n,i]))				
				sl[n,i] = (g if i==0 else sl[n,i-1]+g)
				dr[n,i] = np.full(shape=self.policy.nParams,fill_value=(self.gamma**i)*eps_r[n,i])
			for j in range(T,maxEpLength):
				sl[n,j] = sl[n,j-1]
			if showProgress:
				bar.next()
		'''

        for n, T in enumerate(eps_len):
            # print(eps_s[n].shape)
            # print(T)
            # print(self.policy.compute_log_gradient(eps_s[n, :T], eps_a[n, :T]).shape)
            g = self.policy.compute_log_gradient(eps_s[n, :T], eps_a[n, :T])
            flat_g = np.reshape(g, (T, -1))
            sl[n, :T] = np.cumsum(flat_g, axis=0)
            dr[n, :T] = self.gamma ** np.arange(T) * eps_r[n, :T]
            if showProgress:
                bar.next()

        if showProgress:
            bar.finish()

        #
        # Compute the baseline
        #

        num = np.sum(sl * sl * dr[:, :, None], axis=0)
        den = np.sum(sl * sl, axis=0) + self.alg_epsilon
        b = num / den

        #
        # Compute the gradient
        #

        grads_linear = sl * (dr[:, :, None] - b[None])
        '''
		gradient_ep_linear = np.sum(grads_linear,axis=1)/nEpisodes
		gradient_linear = np.sum(gradient_ep_linear,axis=0)
		gradient = np.reshape(gradient_linear,newshape=self.policy.paramsShape)
		'''
        gradient_ep = np.sum(grads_linear, axis=1)
        
        if getEstimates:
            return gradient_ep

        if initialIS is None:
            gradient = np.reshape(np.mean(gradient_ep, axis=0), newshape=self.policy.paramsShape)
            return gradient
        else:
            # importance sampling
            #gradient = np.reshape(np.average(gradient_ep,axis=0,weights=initialIS), newshape=self.policy.paramsShape)
            gradient = np.mean(((gradient_ep.T*initialIS).T),axis=0)
            return np.reshape(gradient,newshape=self.policy.paramsShape)

        #
        # Compute the sample variance
        #

        '''
		variance = np.zeros(shape=self.policy.paramsShape, dtype=np.float32)
		for i in range(nEpisodes):
			variance += np.reshape(np.square(gradient_ep_linear[i]-gradient_linear),newshape=self.policy.paramsShape)
		variance = variance/nEpisodes
		'''
        variance = np.var(gradient_ep, axis=0)
        variance = np.reshape(variance, self.policy.paramsShape)

        return (gradient, variance)

    def getFisherInformation(self, data):

        eps_s = data["s"]
        eps_a = data["a"]
        eps_len = data["len"]

        fisherInformation = np.zeros(shape=(self.policy.nParams,self.policy.nParams),dtype=np.float32)

        for n,T in enumerate(eps_len):
            g = self.policy.compute_log_gradient(eps_s[n,:T],eps_a[n,:T])
            flat_g = np.reshape(g, (T, -1))
            f = np.matmul(flat_g.T,flat_g)
            fisherInformation += f

        fisherInformation /= np.sum(eps_len)
        
        '''
        w,v = np.linalg.eig(fisherInformation)
        c = np.linalg.cond(fisherInformation)

        print("eigenvalues: ",w)
        print("cond: ",c)
        '''

        return fisherInformation