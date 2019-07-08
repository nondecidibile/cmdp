from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 50
LEARNING_STEPS = 100
LEARNING_EPISODES = 10

CONFIGURATION_STEPS = 100

MAX_NUM_TRIALS = 3
N = 30 # number of episodes collected for the LR test and the configuration