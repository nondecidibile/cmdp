from gym.envs.classic_control import minigolf
from util.util_minigolf import *
from util.policy_gaussian import GaussianPolicy
from util.learner import *
from util.optimizer import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

sfMask = np.array([1,1,0,1,0,0], dtype=np.bool)

'''
range_size = 17
putter_lengths = np.linspace(2,10,range_size)
rewards = np.zeros(range_size)
'''

LEARNING_STEPS = 0
LEARNING_EPS = 1000
LEARNING_RATE = 0.001

N = 10000


#for exp_i in range(range_size):

mdp = minigolf.MiniGolfConf()
#mdp.putter_length = putter_lengths[exp_i]
mdp.putter_length = 1.0
mdp.sigma_noise = 0.01

policy = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
policy.covarianceMatrix = 0.01 ** 2 * np.eye(1)
#policy.init_random_params(stddev=0.1)
policy.params = np.array([[1.145619, 0.025101, 0.319547]])

learner = GpomdpLearner(mdp,policy,gamma=0.95)
#print("\nPutter length: ",putter_lengths[exp_i],flush=True)

best_r = learn(
	learner = learner,
	steps = LEARNING_STEPS,
	nEpisodes = LEARNING_EPS,
	sfmask=sfMask,
	learningRate = LEARNING_RATE,
	plotGradient = True,
	printInfo = True
)

eps = collect_minigolf_episodes(learner.mdp,learner.policy,N,learner.mdp.horizon,sfmask=sfMask,showProgress=True)
r = np.mean(np.sum(eps["r"], axis=1))
'''
r = 0
for n,T in enumerate(eps["len"]):
	r += np.sum(learner.gamma ** np.arange(T) * eps["r"][n, :T])
r /= N
'''
print("Best params: ",repr(learner.policy.params),flush=True)
print("Mean reward: ",r)

#rewards[exp_i] = r

'''
print("\n---\n")
print("Putter lengths: ",repr(putter_lengths))
print("Mean rewards: ",repr(rewards))
'''
