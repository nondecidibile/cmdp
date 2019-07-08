from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

mdp = car_conf.ConfDrivingEnv(model_w = 0.1, renderFlag=True)
mdp.horizon = 100

'''
mdp.reset()
for i in range(100):
	mdp.step([1.0,0])
	mdp.render()
'''

policy = nnGaussianPolicy(nStateFeatures=12,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.25)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

learn(
	learner = learner,
	steps = 25,
	nEpisodes = 10,
	sfmask = None,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

eps = collect_car_episodes(mdp,policy,100,mdp.horizon,sfmask=None,render=False,showProgress=True)

for i in range(100):
	sfGradientMask = np.zeros(shape=12,dtype=np.bool)
	sfGradientMask[0] = True
	g = getModelGradient(learner,eps, sfTarget=0, model_w_new=15, model_w=10)
	mdp = car_conf.ConfDrivingEnv(model_w = mdp.model_w+0.01*g, renderFlag=True)
	print(mdp.model_w)