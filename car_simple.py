from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = car_conf.ConfDrivingEnv(model_w = 0.1, renderFlag=True)
mdp.horizon = 100

policy = nnGaussianPolicy(nStateFeatures=12,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.1)
learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

params = np.load("car_params.npy")

learn(
	learner = learner,
	steps = 0,
	initParams=params,
	nEpisodes = 10,
	sfmask=np.array([1,1,1,1,1,1,1,1,1,1,1,1],dtype=np.bool),
	learningRate = 0.003,
	plotGradient = False,
	printInfo = True
)

#np.save("car_params_1.npy",policy.get_params())

N = 250

eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

lrTest(eps,policy,np.array([0,0,1,1,1,1,1,1,1,1,1,1],dtype=np.bool),numResets=5)

'''
estimated_params = policy.estimate_params(eps,lr=0.01,batchSize=100)
ll = policy.getLogLikelihood(eps)

lr_lambda_0 = -2*(ll0 - ll)
lr_lambda_1 = -2*(ll1 - ll)
print("lr lambda: ",lr_lambda_0,lr_lambda_1)
x = chi2.ppf(0.99,policy.nHiddenNeurons)
print("chi2: ",x)
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