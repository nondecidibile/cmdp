from gym.envs.classic_control import minigolf
from util.util_minigolf import *
from util.policy_gaussian import GaussianPolicy
from util.learner import *
from util.optimizer import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

#sfMask = np.array([1,1,1,1,1,1], dtype=np.bool)
sfMask = np.array([1,1,0,1,0,0], dtype=np.bool)

mdp = minigolf.MiniGolfConf()
mdp.putter_length = 6
mdp.sigma_noise = 0.01

params = np.array([[ 0.0153449, -0.0003801,  0.0054212]]) * mdp.putter_length
#params = np.array([[0, 0, 0, 0, 0, 0.7487/5]]) * mdp.putter_length

policy = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
policy.covarianceMatrix = 0.01 ** 2 * np.eye(1)
policy.params = params

learner = GpomdpLearner(mdp,policy,gamma=0.99)


r = learn(
	learner = learner,
	steps = 1000,
	initParams=params,
	nEpisodes = 50,
	sfmask=sfMask,
	learningRate = 0.001,
	plotGradient = False,
	printInfo = True
)
