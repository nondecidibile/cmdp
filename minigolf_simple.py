from gym.envs.classic_control import minigolf
from util.util_minigolf import *
from util.policy_gaussian import GaussianPolicy
from util.learner import *
from util.optimizer import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

sfMask = np.array([1,1,0,1,0,0], dtype=np.bool)

mdp = minigolf.MiniGolfConf()
mdp.putter_length = 3
mdp.sigma_noise = 0.01

params = np.array([[0.511686, -0.005523, 0.156221]]) # putter_length = 3
#params = np.array([[0.275924, -0.00313,   0.085145]]) # putter_length = 6

policy = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
policy.covarianceMatrix = 0.01 ** 2 * np.eye(1)
policy.params = params

learner = GpomdpLearner(mdp,policy,gamma=0.99)

r = learn(
	learner = learner,
	steps = 250,
	initParams=params,
	nEpisodes = 500,
	sfmask=sfMask,
	learningRate = 0.001,
	plotGradient = False,
	printInfo = True
)

print(policy.params)

'''
N = 100
eps = collect_minigolf_episodes(mdp,policy,N,mdp.horizon,sfmask=sfMask,showProgress=True,exportAllStateFeatures=True)
lrlambda = lrTest(eps,np.array([1,0,0,0,0,0]))

super_policy = GaussianPolicy(nStateFeatures=6,actionDim=1)
super_policy.covarianceMatrix = 0.01 ** 2 * np.eye(1)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.99)

model_w = 6.0
model_w_new = 6.001
optimizer = AdamOptimizer(1,learning_rate=0.001)
for conf_index in range(100):
	g = getModelGradient(super_learner,eps,N,sfTarget=0,model_w_new=model_w_new,model_w=model_w)
	model_w_new += optimizer.step(np.clip(g,-1e+09,1e+09))
	print("g =",g,"- model_w_new =",model_w_new)
'''