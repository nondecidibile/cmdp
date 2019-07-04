from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

mdp = car_conf.ConfDrivingEnv(model_w = 50, renderFlag=True)
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
	steps = 1000,
	nEpisodes = 10,
	sfmask = None,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

ep = collect_car_episode(mdp,policy,mdp.horizon,sfmask=None,render=True)
states = ep[0]["s"]
actions = ep[0]["a"]

print(mdp.p_model(states[1][0:7],states[0],actions[0],50))
print(mdp.grad_log_p_model(states[1][0:7],states[0],actions[0],50))