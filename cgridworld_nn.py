from gym.envs.toy_text import gridworld_cont_normal
from util.util_cgridworld import *
from util.policy_nn import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = gridworld_cont_normal.GridworldContNormalEnv()
mdp.horizon = 30
agent_policy = NeuralNetworkPolicy(nStateFeatures=50,actionDim=2,nHiddenNeurons=16,gamma=0.98)

clearn_nn(
	mdp,
	agent_policy,
	steps=1000,
	nEpisodes=50,
	learningRate=0.1,
	plotGradient=True,
	printInfo=True
)