from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = car_conf.ConfDrivingEnv(model_w = 100, renderFlag=True)
mdp.horizon = 100

policy = nnGaussianPolicy(nStateFeatures=4,actionDim=2,nHiddenNeurons=4,paramInitMaxVal=0.01,variance=0.1)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

params = np.load("car_params_0101.npy")

sfMask = np.array([0,1,0,1],dtype=np.bool)

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

#np.save("car_params_0101.npy",policy.get_params())

N = 500

eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

sfTestMask = np.array([0,0,0,0],dtype=np.bool)
lr_lambda = lrTest(
	eps=eps,
	policyInstance=policy,
	sfMask=sfTestMask,
	nsf=4,
	na=2,
	lr=0.003,
	batchSize=N,
	epsilon=0.001,
	maxSteps=25000,
	numResets=1
)

print("REAL AGENT MASK",sfMask)
print("ESTIMATED AGENT MASK",sfTestMask)