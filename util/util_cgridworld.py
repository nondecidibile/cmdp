import numpy as np
from util.optimizer import *
from progress.bar import Bar
import matplotlib.pyplot as plt
from tkinter import *
import time

def build_cgridworld_features(mdp,state,stateFeaturesMask=None):

	D = mdp.DIM
	ds = D/5
	sigma = 1

	x,y,xg,yg = state
	p = np.array([x,y],dtype=np.float32)
	p_goal = np.array([xg,yg],dtype=np.float32)
	state_features = []

	for j in range(-2,3):
		for i in range(-2,3):
			grid_point = np.array([ds*i,ds*j],dtype=np.float32)
			p_dist = np.linalg.norm(p-grid_point)
			#sf = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.square(p_dist/sigma)/2)
			sf = np.exp(-p_dist**2)
			state_features.append(sf)
	
	for j in range(-2,3):
		for i in range(-2,3):
			grid_point = np.array([ds*i,ds*j],dtype=np.float32)
			p_goal_dist = np.linalg.norm(p_goal-grid_point)
			#sf = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.square(p_goal_dist/sigma)/2)
			sf = np.exp(-p_goal_dist**2)
			state_features.append(sf)
	
	state_features = np.array(state_features, dtype=np.float32)

	if stateFeaturesMask is None:
		return state_features
	else:
		mask = np.array(stateFeaturesMask, dtype=bool)
		#assert(len(mask) == len(state_features))
		return state_features[mask]
			

def draw_circle(x, y, r, canvas, **kwargs):
	x0 = x - r
	y0 = y - r
	x1 = x + r
	y1 = y + r
	return canvas.create_oval(x0, y0, x1, y1, **kwargs)


def lerp(a,b,alpha):
	return b*alpha + (1-alpha)*a


def draw_grid(canvas,DIM,mdp_dim):
	for i in range(-2,3):
		for j in range(-2,3):
			draw_circle(DIM/2+i*(DIM/mdp_dim),DIM-(DIM/2+j*(DIM/mdp_dim)),1,canvas,fill="white")
			if i==-2 and j==-2:
				draw_circle(DIM/2+i*(DIM/mdp_dim),DIM-(DIM/2+j*(DIM/mdp_dim)),2,canvas,fill="yellow")
			if i==2 and j==2:
				draw_circle(DIM/2+i*(DIM/mdp_dim),DIM-(DIM/2+j*(DIM/mdp_dim)),2,canvas,fill="orange")


def collect_cgridworld_episode(mdp,policy,horizon,stateFeaturesMask=None,exportAllStateFeatures=True,render=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)
	adim = policy.actionDim

	states = np.zeros(shape=(horizon,nsf),dtype=np.float32)
	actions = np.zeros(shape=(horizon,adim),dtype=np.float32)
	rewards = np.zeros(shape=horizon,dtype=np.int32)

	state = mdp.reset()

	x,y,xg,yg = state
	if render:
		DIM = 300
		window = Tk()
		canvas = Canvas(window, width=DIM, height=DIM, borderwidth=0, highlightthickness=0, bg="black")
		canvas.pack()
		xx = x
		yy = y

	length = 0

	for i in range(horizon):

		length += 1

		state_features = build_cgridworld_features(mdp,state,stateFeaturesMask)
		action = policy.draw_action(state_features)

		if exportAllStateFeatures is True:
			state_features = build_cgridworld_features(mdp,state)

		newstate, reward, done = mdp.step(action)

		states[i] = state_features
		actions[i] = action
		if done:
			rewards[i] = 0
		else:
			rewards[i] = -1

		if render:
			x,y,xg,yg = state
			for i in range(30):
				tx = lerp(xx,x,i/30)
				ty = lerp(yy,y,i/30)
				canvas.delete('all')
				draw_grid(canvas,DIM,mdp.DIM)
				draw_circle(DIM/2+tx*(DIM/mdp.DIM),DIM-(DIM/2+ty*(DIM/mdp.DIM)),10,canvas,fill="blue")
				draw_circle(DIM/2+xg*(DIM/mdp.DIM),DIM-(DIM/2+yg*(DIM/mdp.DIM)),DIM*mdp.END_DISTANCE/mdp.DIM,canvas,outline="red")
				window.update()
				time.sleep(0.004)
			xx = x
			yy = y
		
		state = newstate

		if done:
			break
		
	
	if render:
		x,y,xg,yg = state
		for i in range(30):
			tx = lerp(xx,x,i/30)
			ty = lerp(yy,y,i/30)
			canvas.delete('all')
			draw_grid(canvas,DIM,mdp.DIM)
			draw_circle(DIM/2+tx*(DIM/mdp.DIM),DIM-(DIM/2+ty*(DIM/mdp.DIM)),10,canvas,fill="blue")
			draw_circle(DIM/2+xg*(DIM/mdp.DIM),DIM-(DIM/2+yg*(DIM/mdp.DIM)),DIM*mdp.END_DISTANCE/mdp.DIM,canvas,outline="red")
			window.update()
			time.sleep(0.004)
		window.destroy()

	episode_data = {"s": states,"a": actions,"r": rewards}
	return [episode_data,length]


def collect_cgridworld_episodes(
	mdp, policy, num_episodes, horizon,
	stateFeaturesMask=None, exportAllStateFeatures=True,
	render=False, showProgress=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)
	adim = policy.actionDim

	data_s = np.zeros(shape=(num_episodes,horizon,nsf),dtype=np.float32)
	data_a = np.zeros(shape=(num_episodes,horizon,adim),dtype=np.float32)
	data_r = np.zeros(shape=(num_episodes,horizon),dtype=np.int32)
	data_len = np.zeros(shape=num_episodes, dtype=np.int32)
	data = {"s": data_s, "a": data_a, "r": data_r, "len": data_len}

	mean_length = 0
	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)
	
	for i in range(num_episodes):
		episode_data, length = collect_cgridworld_episode(mdp,policy,horizon,stateFeaturesMask,exportAllStateFeatures,render)
		data["s"][i] = episode_data["s"]
		data["a"][i] = episode_data["a"]
		data["r"][i] = episode_data["r"]
		data["len"][i] = length
		if showProgress:
			bar.next()

	if showProgress:
		bar.finish()	

	return data


def clearn(learner, steps, nEpisodes, sfmask=None, adamOptimizer=True, learningRate=0.1,
		loadFile=None, saveFile=None, autosave=False, plotGradient=False):

	if loadFile is not None:
		learner.policy.params = np.load(loadFile)
	
	if steps<=0 or steps is None:
		plotGradient = False
	
	if plotGradient:
		xs = []
		mys = []
		ys = []
		plt.plot(xs,ys)
		plt.plot(xs,mys)

	if adamOptimizer:
		optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=learningRate, beta1=0.9, beta2=0.99)

	avg = 0.95
	avg_mean_length = 0
	avg_max_gradient = 0
	avg_mean_gradient = 0
	mt = avg

	for step in range(steps):

		eps = collect_cgridworld_episodes(
			learner.mdp,learner.policy,nEpisodes,learner.mdp.horizon,
			stateFeaturesMask=sfmask,exportAllStateFeatures=False)

		gradient = learner.estimate_gradient(eps)
		if adamOptimizer:
			update_step = optimizer.step(gradient)
		else:
			update_step = gradient*learningRate

		learner.policy.params += update_step

		print("Step: "+str(step))
		mean_length = np.mean(eps["len"])
		avg_mean_length = avg_mean_length*avg+mean_length*(1-avg)
		avg_mean_length_t = avg_mean_length/(1-mt)
		print("Mean length: "+str(np.round(mean_length,3)))
		#print("Average mean length: "+str(np.round(avg_mean_length_t,3)))
		max_gradient = np.max(np.abs(gradient))
		avg_max_gradient = avg_max_gradient*avg+max_gradient*(1-avg)
		avg_max_gradient_t = avg_max_gradient/(1-mt)
		print("Maximum gradient: "+str(np.round(max_gradient,5)))
		#print("Avg maximum gradient: "+str(np.round(avg_max_gradient_t,5)))
		mean_gradient = np.mean(np.abs(gradient))
		avg_mean_gradient = avg_mean_gradient*avg+mean_gradient*(1-avg)
		avg_mean_gradient_t = avg_mean_gradient/(1-mt)
		mt = mt*avg
		print("Mean gradient: "+str(np.round(mean_gradient,5)))
		#print("Avg mean gradient: "+str(np.round(avg_mean_gradient_t,5))+"\n")

		if plotGradient:
			xs.append(step)
			ys.append(mean_gradient)
			mys.append(avg_mean_gradient_t)
			plt.gca().lines[0].set_xdata(xs)
			plt.gca().lines[0].set_ydata(ys)
			plt.gca().lines[1].set_xdata(xs)
			plt.gca().lines[1].set_ydata(mys)
			plt.gca().relim()
			plt.gca().autoscale_view()
			plt.yscale("log")
			plt.pause(0.01)

		if saveFile is not None and autosave and step%10==0:
			np.save(saveFile,learner.policy.params)
			print("Params saved in ",saveFile,"\n")

	if saveFile is not None:
		np.save(saveFile,learner.policy.params)
		print("Params saved in ",saveFile,"\n")
	