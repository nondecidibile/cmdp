import numpy as np
import time
from gym.envs.toy_text import taxi, gridworld
from util.optimizer import *
from progress.bar import Bar
import matplotlib.pyplot as plt


def onehot_encode(i, n):
	action = []
	for j in range(n):
		action.append(int(i==j))
	return action


def build_taxi_features(mdp,observation):

	lst = list(mdp.decode(observation))
	pos_x, pos_y = lst[1], lst[0]

	if lst[2]<4:
		src = mdp.locs[lst[2]]
	else:
		src = [pos_x, pos_y]
	dst = mdp.locs[lst[3]]
	state_features = []

	# taxi position
	state_features += onehot_encode(pos_x,4)
	state_features += onehot_encode(pos_y,4)
	# src position
	# ASSUMPTION: if passenger is on board src position = taxi position
	state_features += onehot_encode(src[0],4)
	state_features += onehot_encode(src[1],4)
	# dst position
	state_features += onehot_encode(dst[0],4)
	state_features += onehot_encode(dst[1],4)
	# passenger on board
	state_features += int(lst[2]==4)

	return state_features


def build_gridworld_features(mdp,observation,stateFeaturesMask=None):

	lst = list(mdp.decode(observation))
	row, col, row_g, col_g = lst[0], lst[1], lst[2], lst[3]

	state_features = []

	# taxi position
	state_features += onehot_encode(row,4)
	state_features += onehot_encode(col,4)
	# dst position
	state_features += onehot_encode(row_g,4)
	state_features += onehot_encode(col_g,4)

	if stateFeaturesMask is None:
		return state_features
	else:
		state_features = np.array(state_features)
		mask = np.array(stateFeaturesMask, dtype=bool)
		return state_features[mask]


def collect_gridworld_episode(mdp,policy,horizon,stateFeaturesMask=None,exportAllStateFeatures=True,render=False):

	states = []
	actions = []
	rewards = []

	state = mdp.reset()
	if render:
		mdp.render()

	length = 0

	for i in range(horizon):

		length += 1

		state_features = build_gridworld_features(mdp,state,stateFeaturesMask)
		action = policy.draw_action(state_features)

		if exportAllStateFeatures is True:
			state_features = build_gridworld_features(mdp,state)

		newstate, reward, done, _ = mdp.step(action)
		states.append(state_features)
		actions.append(action)
		if done:
			rewards.append(0)
		else:
			rewards.append(-1)

		if render:
			mdp.render()
			time.sleep(0.1)

		if done:
			break
		
		state = newstate
	
	episode_data = {"s": states,
			"a": np.array(actions,dtype=np.int32),
			"r": np.array(rewards,dtype=np.int32)}
	return [episode_data,length]


def collect_gridworld_episodes(mdp,policy,num_episodes,horizon,stateFeaturesMask=None,exportAllStateFeatures=True,render=False,showProgress=False):
	
	data = []

	mean_length = 0
	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)

	for i in range(num_episodes):
		episode_data, length = collect_gridworld_episode(mdp,policy,horizon,stateFeaturesMask,exportAllStateFeatures,render)
		data.append(episode_data)
		mean_length += length
		if showProgress:
			bar.next()

	if showProgress:
		bar.finish()	
	mean_length /= num_episodes

	return [data,mean_length]


def learn(learner, steps, nEpisodes, sfmask, loadFile=None,
		saveFile=None, autosave=False, plotGradient=False):

	if loadFile is not None:
		learner.policy.params = np.load(loadFile)
	
	if steps<=0 or steps is None:
		plotGradient = False
	
	if plotGradient:
		xs = []
		ys = []
		plt.plot(xs,ys)

	optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=0.1, beta1=0.9, beta2=0.99)

	avg = 0.99
	avg_mean_length = 0
	avg_gradient = 0
	mt = avg

	for step in range(steps):

		eps, mean_length = collect_gridworld_episodes(learner.mdp,learner.policy,nEpisodes,
						   learner.mdp.horizon,sfmask,False)

		gradient = learner.estimate_gradient(eps)
		update_step = optimizer.step(gradient)

		learner.policy.params += update_step

		print("Step: "+str(step))
		avg_mean_length = avg_mean_length*avg+mean_length*(1-avg)
		avg_mean_length_t = avg_mean_length/(1-mt)
		print("Average mean length:      "+str(np.round(avg_mean_length_t,3)))
		avg_gradient = avg_gradient*avg+gradient/learner.policy.nFeatures*(1-avg)
		avg_gradient_t = np.linalg.norm(np.ravel(np.asarray(avg_gradient)))/(1-mt)
		mt = mt*avg
		print("Average gradient/param:   "+str(np.round(avg_gradient_t,5))+"\n")
		#print("Update step /param:       "+str(np.round(np.linalg.norm(np.ravel(np.asarray(update_step/learner.policy.nFeatures)),1),5))+"\n")

		if plotGradient:
			xs.append(step)
			ys.append(avg_gradient_t)
			plt.gca().lines[0].set_xdata(xs)
			plt.gca().lines[0].set_ydata(ys)
			plt.gca().relim()
			plt.gca().autoscale_view()
			plt.pause(0.01)

		if saveFile is not None and autosave and step%10==0:
			np.save(saveFile,learner.policy.params)
			print("Params saved in ",saveFile,"\n")

	if saveFile is not None:
		np.save(saveFile,learner.policy.params)
		print("Params saved in ",saveFile,"\n")
	
	if plotGradient:
		plt.show()


def find_params_ml(estLearner,saveFile=None):

	print("Estimating policy parameters with Maximum Likelihood...\n")
	optimizer = AdamOptimizer(estLearner.policy.paramsShape,learning_rate=2.5)
	est_params = estLearner.policy.estimate_params(eps,optimizer,estLearner.policy.params,epsilon=0.01,minSteps=50,maxSteps=200)
		
	#_,est_ml = collect_gridworld_episodes(mdp,estLearner.policy,500,mdp.horizon)
	#print("Estimated avg mean length: ",est_ml,"\n")
	#p_abs_mean = np.mean(np.abs(estLearner.policy.params),axis=0)
	#print("Parameters absolute average between actions\n",p_abs_mean,"\n")

	if saveFile is not None:
		np.save(saveFile,est_params)