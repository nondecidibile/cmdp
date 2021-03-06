import numpy as np
import scipy as sp
from gym.envs.toy_text import gridworld_cont_normal
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mean_model1 = [-2,-2,2,2]
var_model1 = [0.1,0.1,0.1,0.1]
mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=mean_model1,var=var_model1)
mdp.horizon = 50

mdp_uniform = gridworld_cont.GridworldContEnv()
mdp_uniform.horizon = 50

sfMask = np.ones(shape=50,dtype=np.bool) # state features mask
sfMask[45:50] = False

# AGENT SPACE
agent_policy = GaussianPolicy(np.count_nonzero(sfMask),2)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

clearn(
	agent_learner,
	steps=100,
	nEpisodes=250,
	sfmask=sfMask,
	adamOptimizer=True,
	learningRate=0.03,
	loadFile=None,
	saveFile=None,
	autosave=True,
	plotGradient=False
)

# SUPERVISOR SPACE
super_policy = GaussianPolicy(50,2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

# collect episodes with agent's policy
N = 5000
eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
eps_uniform = collect_cgridworld_episodes(mdp_uniform,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)

# estimated agent params and gradient
optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
estimated_params = super_learner.policy.estimate_params(eps_uniform,optimizer,None,epsilon=0.001,minSteps=150,maxSteps=300)

# hypothesis testing on parameters == 0
fisherInfo = super_learner.policy.getAnalyticalFisherInformation(eps_uniform)
invFisherDiag = np.diagonal(np.linalg.inv(fisherInfo))
var_terms = np.ravel(np.sqrt(invFisherDiag))
estimator_terms = super_learner.policy.params * np.sqrt(N) / var_terms
sfTestMask = np.ones(shape=50,dtype=np.bool) # State features not rejected
z_0_90 = 1.65
z_0_95 = 1.96
z_0_99 = 2.58
for a,tt in enumerate(estimator_terms):
	print("\n",end='')
	for s,t in enumerate(tt):
		rejected90 = 0
		rejected95 = 0
		rejected99 = 0
		p = np.abs(t)
		if not(p+z_0_90>=0 and p-z_0_90<=0):
			rejected90 = 1
		if not(p+z_0_95>=0 and p-z_0_95<=0):
			rejected95 = 1
			sfTestMask[s] = False
		if not(p+z_0_99>=0 and p-z_0_99<=0):
			rejected99 = 1
		print("s =",s,"| a =",a,
		"|| p = ",format(p,".5f"),
		"| rejected (@90%, 95%, 99%) =",rejected90,rejected95,rejected99)

sfTestMaskTiled = np.tile(sfTestMask,(2,1))
sfTestMaskLin = np.ravel(sfTestMaskTiled)
print(sfTestMaskTiled[0])

for __i in range(100):

	# importance sampling
	initialStates = eps["state"][:,0]
	N1 = sp.stats.multivariate_normal(mean=mean_model1,cov=np.diag(var_model1))

	mean_model2 = mean_model1.copy()
	mean_model2[0] += 0.1
	mean_model2[2] -= 0.1
	var_model2 = var_model1.copy()

	model_optimizer = AdamOptimizer(shape=4,learning_rate=0.01)

	for _i in range(150):
		N2 = sp.stats.multivariate_normal(mean=mean_model2,cov=np.diag(var_model2))
		initialIS = N2.pdf(initialStates) / N1.pdf(initialStates)
		#print(np.linalg.norm(initialIS,ord=np.inf))
		estimated_gradient2 = super_learner.estimate_gradient(eps,initialIS)
		#print("MODEL 2:\n",estimated_gradient2)

		# norm^2_2 of the gradient of J
		x = np.linalg.norm(estimated_gradient2[sfTestMaskTiled])**2

		grads_estimates = super_learner.estimate_gradient(eps,getEstimates=True)
		dj = np.ravel(estimated_gradient2)

		ddj = np.zeros(shape=(len(eps["len"]),len(mean_model2),len(sfTestMaskLin)),dtype=np.float32)
		for n in range(len(eps["len"])):
			hx1 = (initialStates[n]-mean_model2)/var_model2
			kx1 = grads_estimates[n]
			ddj[n] = np.outer(hx1,kx1)
		ddj = (ddj.T*initialIS).T
		ddj = np.sum(ddj,axis=0)/len(eps["len"])

		dj = dj[sfTestMaskLin]
		ddj = ddj[:,sfTestMaskLin]
		model_gradient_J = np.matmul(ddj,dj)

		# ranyi divergence term
		lambda_param = 1/4 * np.linalg.norm(estimated_gradient2,ord=np.inf)*np.sqrt(np.count_nonzero(sfTestMaskLin)/0.95)
		print("lambda =",lambda_param)
		print("d2 = ",d2gaussians(mean_model2,np.diag(var_model2),mean_model1,np.diag(var_model1)))
		print("weights = ",np.mean(initialIS**2))

		model_gradient_div = lambda_param/2/np.sqrt(len(eps["len"]))
		model_gradient_div *= d_d2gaussians_dmuP(mean_model2,np.diag(var_model2),mean_model1,np.diag(var_model1))
		model_gradient_div /= np.sqrt(d2gaussians(mean_model2,np.diag(var_model2),mean_model1,np.diag(var_model1)))

		model_gradient = model_gradient_J - model_gradient_div
		print("norm_2^2(grad J) =",x)
		print("model gradient [J] =",model_gradient_J)
		print("model gradient [div] =",model_gradient_div)
		print("model gradient =",model_gradient)
		update_step = model_optimizer.step(model_gradient)
		mean_model2 += update_step
		print("new model =",mean_model2,"\n")



	# use new model

	mean_model1 = mean_model2.copy()
	var_model1 = var_model2.copy()

	mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=mean_model1,var=var_model1)
	mdp.horizon = 50

	mdp_uniform = gridworld_cont.GridworldContEnv()
	mdp_uniform.horizon = 50

	# AGENT SPACE
	agent_policy = GaussianPolicy(np.count_nonzero(sfMask),2)
	agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

	clearn(
		agent_learner,
		steps=100,
		nEpisodes=250,
		sfmask=sfMask,
		adamOptimizer=True,
		learningRate=0.03,
		loadFile=None,
		saveFile=None,
		autosave=True,
		plotGradient=False
	)

	# SUPERVISOR SPACE
	super_policy = GaussianPolicy(50,2)
	super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

	# collect episodes with agent's policy
	N = 5000
	eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
	eps_uniform = collect_cgridworld_episodes(mdp_uniform,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)

	# estimated agent params and gradient
	optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
	estimated_params = super_learner.policy.estimate_params(eps_uniform,optimizer,None,epsilon=0.001,minSteps=150,maxSteps=500)

	# hypothesis testing on parameters == 0
	fisherInfo = super_learner.policy.getAnalyticalFisherInformation(eps)
	invFisherDiag = np.diagonal(np.linalg.inv(fisherInfo))
	var_terms = np.ravel(np.sqrt(invFisherDiag))
	estimator_terms = super_learner.policy.params * np.sqrt(N) / var_terms
	z_0_90 = 1.65
	z_0_95 = 1.96
	z_0_99 = 2.58
	#sfTestMask = np.ones(shape=50,dtype=np.bool) # State features not rejected
	for a,tt in enumerate(estimator_terms):
		print("\n",end='')
		for s,t in enumerate(tt):
			if sfTestMask[s]:
				rejected90 = 0
				rejected95 = 0
				rejected99 = 0
				p = np.abs(t)
				if not(p+z_0_90>=0 and p-z_0_90<=0):
					rejected90 = 1
				if not(p+z_0_95>=0 and p-z_0_95<=0):
					rejected95 = 1
					sfTestMask[s] = False
				if not(p+z_0_99>=0 and p-z_0_99<=0):
					rejected99 = 1
				print("s =",s,"| a =",a,
				"|| p = ",format(p,".5f"),
				"| rejected (@90%, 95%, 99%) =",rejected90,rejected95,rejected99)
	sfTestMaskTiled = np.tile(sfTestMask,(2,1))
	sfTestMaskLin = np.ravel(sfTestMaskTiled)
	print(sfTestMaskTiled[0])
