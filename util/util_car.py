import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
from util.optimizer import *


def build_state_features(state,sfmask=None):
	x = np.array(state)
	if sfmask is not None:
		mask = np.array(sfmask, dtype=bool)
		x[1-mask] = 0
	return x


def collect_car_episode(mdp,policy,horizon,sfmask=None,render=False):

	states = np.zeros(shape=(horizon,policy.nStateFeatures),dtype=np.float32)
	actions = np.zeros(shape=(horizon,2),dtype=np.float32)
	rewards = np.zeros(shape=horizon,dtype=np.float32)

	state = mdp.reset()
	state = build_state_features(state,sfmask)
	if render:
		mdp.render()

	length = 0

	for i in range(horizon):

		length += 1

		action = policy.draw_action([state])
		newstate, normSpeed, offroad, done = mdp.step(action)

		states[i] = state
		actions[i] = action
		rewards[i] = normSpeed if (not offroad) else -25

		if render:
			mdp.render()
		
		state = build_state_features(newstate,sfmask)

		if done:
			break
	
	if render:
		mdp.render()
	
	episode_data = {"s": states,"a": actions,"r": rewards}
	return [episode_data,length]


def collect_car_episodes(mdp,policy,num_episodes,horizon,sfmask=None,render=False,showProgress=False):
	
	data_s = np.zeros(shape=(num_episodes,horizon,policy.nStateFeatures),dtype=np.float32)
	data_a = np.zeros(shape=(num_episodes,horizon,2),dtype=np.float32)
	data_r = np.zeros(shape=(num_episodes,horizon),dtype=np.float32)
	data_len = np.zeros(shape=num_episodes, dtype=np.int32)
	data = {"s": data_s, "a": data_a, "r": data_r, "len": data_len}

	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)
	
	for i in range(num_episodes):
		episode_data, length = collect_car_episode(mdp,policy,horizon,sfmask,render)
		data["s"][i] = episode_data["s"]
		data["a"][i] = episode_data["a"]
		data["r"][i] = episode_data["r"]
		data["len"][i] = length
		if showProgress:
			bar.next()

	if showProgress:
		bar.finish()	

	return data


def learn(learner, steps, nEpisodes, sfmask=None, learningRate=0.1, plotGradient=False, printInfo=False):
	
	if steps<=0 or steps is None:
		plotGradient = False
	
	if plotGradient:
		xs = []
		ys1 = []
		ys2 = []
		plt.subplot(2, 1, 1)
		plt.plot(xs,ys1)
		plt.ylabel('Mean reward')
		plt.subplot(2, 1, 2)
		plt.plot(xs,ys2)
		plt.ylabel('Mean gradient')
	
	learner.policy.s.run(learner.policy.init_op)
	optimizer = AdamOptimizer(learner.policy.nParams, learning_rate=learningRate, beta1=0.9, beta2=0.99)

	avg = 0.95
	avg_mean_length = 0
	avg_max_gradient = 0
	avg_mean_gradient = 0
	mt = avg

	print("step, mean_reward, max_gradient, mean_update")

	for step in range(steps):

		eps = collect_car_episodes(learner.mdp,learner.policy,nEpisodes,learner.mdp.horizon,sfmask,render=False)
		gradient = learner.optimize_gradient(eps,optimizer)

		mean_length = np.mean(eps["len"])
		avg_mean_length = avg_mean_length*avg+mean_length*(1-avg)
		avg_mean_length_t = avg_mean_length/(1-mt)
		#print("Average mean length: "+str(np.round(avg_mean_length_t,3)))
		max_gradient = np.max(np.abs(gradient))
		avg_max_gradient = avg_max_gradient*avg+max_gradient*(1-avg)
		avg_max_gradient_t = avg_max_gradient/(1-mt)
		#print("Avg maximum gradient: "+str(np.round(avg_max_gradient_t,5)))
		mean_gradient = np.mean(np.abs(gradient))
		avg_mean_gradient = avg_mean_gradient*avg+mean_gradient*(1-avg)
		avg_mean_gradient_t = avg_mean_gradient/(1-mt)
		#print("Avg mean gradient: "+str(np.round(avg_mean_gradient_t,5))+"\n")
		mean_reward = np.mean(eps["r"])
		mt = mt*avg
		if printInfo:
			'''print("Step: "+str(step))
			print("Mean length: "+str(np.round(mean_length,3)))
			print("Mean gradient: "+str(np.round(mean_gradient,5)))
			print("Maximum gradient: "+str(np.round(max_gradient,5)))
			print("Mean rewards: ",str(np.round(mean_reward,5)))
			print("Mean update: ",str(np.round(mean_gradient*learningRate,5)))
			'''
			print(str(step),",  ",str(np.round(mean_reward,5)),",  ",str(np.round(max_gradient,5)),",   ",str(np.round(mean_gradient*learningRate,5)))

		if plotGradient:
			xs.append(step)
			ys1.append(mean_reward)
			ys2.append(mean_gradient)
			plt.subplot(2, 1, 1)
			plt.gca().lines[0].set_xdata(xs)
			plt.gca().lines[0].set_ydata(ys1)
			plt.gca().relim()
			plt.gca().autoscale_view()
			plt.subplot(2, 1, 2)
			plt.gca().lines[0].set_xdata(xs)
			plt.gca().lines[0].set_ydata(ys2)
			plt.gca().relim()
			plt.gca().autoscale_view()
			#plt.yscale("log")
			plt.pause(0.000001)
		
		#policy.print_params()
		if step>0 and step%5==0:
			collect_car_episode(learner.mdp,learner.policy,learner.mdp.horizon,sfmask=sfmask,render=True)