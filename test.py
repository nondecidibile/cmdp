import numpy as np
import scipy as sp
from gym.envs.toy_text import gridworld_cont_normal
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mean_model1 = [-2,-2,2,2]
var_model1 = [0.1,0.1,0.1,0.1]
mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=mean_model1,var=var_model1)

initialStates = []
for i in range(10000):
	x,y,xg,yg = mdp.reset()
	initialStates.append(np.array([x,y,xg,yg]))


N1 = sp.stats.multivariate_normal(mean=mean_model1,cov=np.diag(var_model1))

mean_model2 = [-1.9,-1.9,1.9,1.9]
var_model2 = [0.1,0.1,0.1,0.1]

N2 = sp.stats.multivariate_normal(mean=mean_model2,cov=np.diag(var_model2))
initialIS = N2.pdf(initialStates) / N1.pdf(initialStates)

print("d2 = ",d2gaussians(mean_model2,np.diag(var_model2),mean_model1,np.diag(var_model1)))
print("weights = ",np.mean(initialIS**2))