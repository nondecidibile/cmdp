from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = car_conf.ConfDrivingEnv(model_w = 100, renderFlag=True)
mdp.horizon = 100

policy = nnGaussianPolicy(nStateFeatures=12,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.1)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

params = np.load("car_params_03.npy")

print(policy.s.run(policy.params))

sfMask = np.array([1,0,0,1,0,0,1,0,1,0,1,0],dtype=np.bool)
learn(
	learner = learner,
	steps = 0, # 100 steps with 25 episodes is ok
	initParams=params,
	nEpisodes = 25,
	sfmask=sfMask,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

#np.save("car_params_03.npy",policy.get_params())

N = 250

eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=True,showProgress=True)

'''
ll0 = np.zeros(shape=12,dtype=np.float)
ll_tot = np.zeros(shape=12,dtype=np.float)

for f in range(2):
	params0 = policy.estimate_params(eps,nullFeature=f,lr=0.003,batchSize=N,maxSteps=2500)
	ll0[f] = policy.getLogLikelihood(eps)
	params = policy.estimate_params(eps,params0=params0,lockExcept=f,lr=0.003,batchSize=N,maxSteps=2500)
	ll_tot[f] = policy.getLogLikelihood(eps)

lr_lambda = -2*(ll0 - ll_tot)
print("lr lambda: ",lr_lambda,flush=True)
x = chi2.ppf(0.99,policy.nHiddenNeurons)
print("chi2: ",x,flush=True)
'''
sfTestMask = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.bool)
lr_lambda = lrTest(
	eps=eps,
	policyInstance=policy,
	sfMask=sfTestMask,
	lr=0.003,
	batchSize=N,
	epsilon=0.001,
	maxSteps=10000,
	numResets=1
)

print("REAL AGENT MASK",sfMask)
print("ESTIMATED AGENT MASK",sfTestMask)

'''
params0 = policy.estimate_params(eps,nullFeatures=[0,1,2,3,4,5,6],lr=0.003,batchSize=N,maxSteps=10000)
ll0 = policy.getLogLikelihood(eps)
	
params = policy.estimate_params(eps,nullFeatures=[1,2,3,4,5,6],lr=0.003,batchSize=N,maxSteps=10000)
ll = policy.getLogLikelihood(eps)


print(ll0,ll)
print(-2*(ll0 - ll))
'''

'''
optimizer = AdamOptimizer(1, learning_rate=1.0, beta1=0.9, beta2=0.99)
mdp.model_w += 5
for i in range(250):
	sfGradientMask = np.zeros(shape=12,dtype=np.bool)
	sfGradientMask[0] = True
	g = getModelGradient(learner,eps,25, sfTarget=0, model_w_new=mdp.model_w, model_w=0.1)
	mdp = car_conf.ConfDrivingEnv(model_w = mdp.model_w+optimizer.step(g), renderFlag=True)
	print("NEW MODEL =",mdp.model_w)
'''