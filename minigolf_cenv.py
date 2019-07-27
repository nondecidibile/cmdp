from gym.envs.classic_control import minigolf
from util.util_minigolf import *
from util.policy_gaussian import GaussianPolicy
from util.learner import *
from util.optimizer import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

sfMask = np.array([1,1,1,1,1,1], dtype=np.bool)

range_size = 17
putter_lengths = np.linspace(2,10,range_size)
rewards = np.zeros(range_size)

LEARNING_STEPS = 1000
LEARNING_EPS = 500
LEARNING_RATE = 0.001

N = 25000

for exp_i in range(range_size):

	mdp = minigolf.MiniGolfConf()
	mdp.putter_length = putter_lengths[exp_i]
	mdp.sigma_noise = 0.01

	policy = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
	policy.covarianceMatrix = 0.01 ** 2 * np.eye(1)
	#policy.init_random_params(stddev=0.1)

	if mdp.putter_length<=3:
		policy.params = np.array([[0.229612, 0.026594, 0.559805, 0.052526, 0.468982, 0.332139]]) # 3
	elif mdp.putter_length<=4:
		policy.params = np.array([[-0.002192, 0.005696, 0.33819, 0.062551, 0.292918, 0.65935]]) # 4
	elif mdp.putter_length<=5:
		policy.params = np.array([[-0.036212, 0.004081, 0.242721, 0.044, 0.213129, 0.574209]]) # 5
	elif mdp.putter_length<=6:
		policy.params = np.array([[-0.035552, 0.002448, 0.140966, 0.031859, 0.146986, 0.519569]]) # 6
	elif mdp.putter_length<=7:
		policy.params = np.array([[-0.024849, 0.000364, 0.070778, 0.02272, 0.104006, 0.471554]]) # 7
	elif mdp.putter_length<=8:
		policy.params = np.array([[-0.016004, -0.008742, 0.056379, 0.011282, 0.099697, 0.456323]]) # 8
	elif mdp.putter_length<=9:
		policy.params = np.array([[-0.027855, -0.020536, 0.053943, 0.000836, 0.094026, 0.450082]]) # 9
	else:
		policy.params = np.array([[-0.036755, -0.029334, 0.050084, -0.005624, 0.087806, 0.444036]]) # 10

	learner = GpomdpLearner(mdp,policy,gamma=0.95)

	print("\nPutter length: ",putter_lengths[exp_i],flush=True)

	best_r = learn(
		learner = learner,
		steps = LEARNING_STEPS,
		nEpisodes = LEARNING_EPS,
		sfmask=sfMask,
		learningRate = LEARNING_RATE,
		plotGradient = False,
		printInfo = False
	)

	eps = collect_minigolf_episodes(learner.mdp,learner.policy,N,learner.mdp.horizon,sfmask=sfMask,showProgress=False)
	r = np.mean(np.sum(eps["r"], axis=1))
	print("Best params: ",repr(learner.policy.params),flush=True)
	print("Mean reward: ",r)

	rewards[exp_i] = r

print("\n---\n")
print("Putter lengths: ",repr(putter_lengths))
print("Mean rewards: ",repr(rewards))