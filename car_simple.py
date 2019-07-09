from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

mdp = car_conf.ConfDrivingEnv(model_w = 0.1, renderFlag=False)
mdp.horizon = 100

policy = nnGaussianPolicy(nStateFeatures=12,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.25)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

learn(
	learner = learner,
	steps = 0,
	nEpisodes = 10,
	sfmask = None,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

params = np.load("car_params.npy")
policy.set_params(params)

eps = collect_car_episodes(mdp,policy,25,mdp.horizon,sfmask=None,render=False,showProgress=True)

mdp.model_w += 1
for i in range(250):
	sfGradientMask = np.zeros(shape=12,dtype=np.bool)
	sfGradientMask[0] = True
	g = getModelGradient(learner,eps, sfTarget=0, model_w_new=mdp.model_w, model_w=0.1)
	mdp = car_conf.ConfDrivingEnv(model_w = mdp.model_w+np.clip(0.03*g,-5,5), renderFlag=True)
	print("NEW MODEL =",mdp.model_w)