from gym.envs.classic_control import car
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

mdp = car.DrivingEnv(renderFlag=True)
mdp.horizon = 250

'''
mdp.reset()
for i in range(250):
	mdp.step([1.0,0])
	mdp.render()
'''

policy = nnGaussianPolicy(nStateFeatures=11,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.25)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

learn(
	learner = learner,
	steps = 1000,
	nEpisodes = 10,
	sfmask = None,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)