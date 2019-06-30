import numpy as np
import scipy as sp
from util.optimizer import *
from util.policy_gaussian import *
from progress.bar import Bar
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageDraw
import time
from scipy.stats import chi2

def build_cgridworld_features(mdp,state,transfMatrix=None,stateFeaturesMask=None):

	D = mdp.DIM
	ds = D/5
	sigma = 1

	x,y,xg,yg = state
	p = np.array([x,y],dtype=np.float32)
	p_goal = np.array([xg,yg],dtype=np.float32)
	state_features = []

	i = np.linspace(-2*ds, 2*ds, 5)
	grid_points = np.meshgrid(i,i)

	dist_x = x - grid_points[0]
	dist_y = y - grid_points[1]

	dists = np.ravel(np.linalg.norm([dist_x,dist_y],axis=0))
	asf = np.exp(-dists**2)

	g_dist_x = xg - grid_points[0]
	g_dist_y = yg - grid_points[1]

	gdists = np.ravel(np.linalg.norm([g_dist_x,g_dist_y],axis=0))
	gsf = np.exp(-gdists**2)
	
	state_features = np.ravel([asf,gsf])

	if transfMatrix is not None:
		state_features = np.dot(transfMatrix,state_features)

	if stateFeaturesMask is not None:
		mask = np.array(stateFeaturesMask, dtype=bool)
		state_features = state_features[mask]
	
	return state_features
			

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


def collect_cgridworld_episode(mdp,policy,horizon,transfMatrix=None,
		stateFeaturesMask=None,exportAllStateFeatures=True,render=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)
	adim = policy.actionDim

	original_states = np.zeros(shape=(horizon,4),dtype=np.float32)
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

		state_features = build_cgridworld_features(mdp,state,transfMatrix,stateFeaturesMask)
		action = policy.draw_action(state_features)

		if exportAllStateFeatures is True:
			state_features = build_cgridworld_features(mdp,state,transfMatrix)

		newstate, reward, done = mdp.step(action)

		original_states[i] = np.array([x,y,xg,yg])
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

	episode_data = {"state": original_states, "s": states,"a": actions,"r": rewards}
	return [episode_data,length]


def collect_cgridworld_episodes(
	mdp, policy, num_episodes, horizon, transfMatrix=None,
	stateFeaturesMask=None, exportAllStateFeatures=True,
	render=False, showProgress=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)
	adim = policy.actionDim

	data_state = np.zeros(shape=(num_episodes,horizon,4),dtype=np.float32)
	data_s = np.zeros(shape=(num_episodes,horizon,nsf),dtype=np.float32)
	data_a = np.zeros(shape=(num_episodes,horizon,adim),dtype=np.float32)
	data_r = np.zeros(shape=(num_episodes,horizon),dtype=np.int32)
	data_len = np.zeros(shape=num_episodes, dtype=np.int32)
	data = {"state": data_state, "s": data_s, "a": data_a, "r": data_r, "len": data_len}

	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)
	
	for i in range(num_episodes):
		episode_data, length = collect_cgridworld_episode(mdp,policy,horizon,transfMatrix,stateFeaturesMask,exportAllStateFeatures,render)
		data["state"][i] = episode_data["state"]
		data["s"][i] = episode_data["s"]
		data["a"][i] = episode_data["a"]
		data["r"][i] = episode_data["r"]
		data["len"][i] = length
		if showProgress:
			bar.next()

	if showProgress:
		bar.finish()	

	return data


def clearn(learner, steps, nEpisodes, transfMatrix=None, sfmask=None,
		adamOptimizer=True, learningRate=0.1, loadFile=None, saveFile=None,
		autosave=False, plotGradient=False, printInfo=True):

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
			transfMatrix=transfMatrix, stateFeaturesMask=sfmask,exportAllStateFeatures=False)

		gradient = learner.estimate_gradient(eps)
		if adamOptimizer:
			update_step = optimizer.step(gradient)
		else:
			update_step = gradient*learningRate

		learner.policy.params += update_step

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
		mt = mt*avg
		if printInfo:
			print("Step: "+str(step))
			print("Mean length: "+str(np.round(mean_length,3)))
			print("Mean gradient: "+str(np.round(mean_gradient,5)))
			print("Maximum gradient: "+str(np.round(max_gradient,5)))

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
	

def d2gaussians(muP,covP,muQ,covQ):
	muP = np.array(muP)
	covP = np.array(covP)
	muQ = np.array(muQ)
	covQ = np.array(covQ)
	A = 2*covQ-covP
	t1 = (muP-muQ).T.dot(np.linalg.inv(A)).dot(muP-muQ)
	t2 = 0.5*np.log(np.linalg.det(A)*np.linalg.det(covP)/(np.linalg.det(covQ)**2))
	D2 = t1-t2
	return np.exp(D2)

def d_d2gaussians_dmuP(muP,covP,muQ,covQ):
	muP = np.array(muP)
	covP = np.array(covP)
	muQ = np.array(muQ)
	covQ = np.array(covQ)
	invA = np.linalg.inv(2*covQ-covP)
	t1 = d2gaussians(muP,covP,muQ,covQ)
	t2 = (muP-muQ).T.dot(invA) + (muP-muQ).T.dot(invA.T)
	return t1*t2

def getModelGradient(superLearner, eps, sfGradientMask, meanModel, varModel, meanModel2, varModel2):

	sfGradientMaskTiled = np.tile(sfGradientMask,(2,1))
	sfGradientMaskLin = np.ravel(sfGradientMaskTiled)

	# importance sampling
	initialStates = eps["state"][:,0]
	N1 = sp.stats.multivariate_normal(mean=meanModel,cov=np.diag(varModel))
	N2 = sp.stats.multivariate_normal(mean=meanModel2,cov=np.diag(varModel2))
	initialIS = N2.pdf(initialStates) / N1.pdf(initialStates)
	estimated_gradient2 = superLearner.estimate_gradient(eps,initialIS)

	# norm^2_2 of the gradient of J
	#x = np.linalg.norm(estimated_gradient2[sfGradientMaskTiled])**2

	grads_estimates = superLearner.estimate_gradient(eps,getEstimates=True)
	dj = np.ravel(estimated_gradient2)

	ddj = np.zeros(shape=(len(eps["len"]),len(meanModel2),len(sfGradientMaskLin)),dtype=np.float32)
	for n in range(len(eps["len"])):
		hx1 = (initialStates[n]-meanModel2)/varModel2
		kx1 = grads_estimates[n]
		ddj[n] = np.outer(hx1,kx1)
	ddj = (ddj.T*initialIS).T
	ddj = np.sum(ddj,axis=0)/len(eps["len"])

	dj = dj[sfGradientMaskLin]
	ddj = ddj[:,sfGradientMaskLin]
	model_gradient_J = np.matmul(ddj,dj)

	# ranyi divergence term
	lambda_param = 1/1000000 * np.linalg.norm(estimated_gradient2,ord=np.inf)*np.sqrt(np.count_nonzero(sfGradientMaskLin)/0.95)
	#print("lambda =",lambda_param)
	#print("d2 = ",d2gaussians(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel)))
	#print("weights = ",np.mean(initialIS**2))

	model_gradient_div = lambda_param/2/np.sqrt(len(eps["len"]))
	model_gradient_div *= d_d2gaussians_dmuP(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel))
	model_gradient_div /= np.sqrt(d2gaussians(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel)))

	model_gradient = model_gradient_J - model_gradient_div
	#print("norm_2^2(grad J) =",x)
	#print("model gradient [J] =",model_gradient_J)
	#print("model gradient [div] =",model_gradient_div)
	#print("model gradient =",model_gradient)
	return model_gradient

def updateSfBestModels(superLearner,eps,sfBestModels,meanModel,varModel,meanModel2,varModel2):
	
	# importance sampling
	initialStates = eps["state"][:,0]
	N1 = sp.stats.multivariate_normal(mean=meanModel,cov=np.diag(varModel))
	N2 = sp.stats.multivariate_normal(mean=meanModel2,cov=np.diag(varModel2))
	initialIS = N2.pdf(initialStates) / N1.pdf(initialStates)
	estimated_gradient2 = superLearner.estimate_gradient(eps,initialIS)

	for i in range(len(sfBestModels)):
		grad2 = np.max(estimated_gradient2[:,i]**2)
		if grad2 > sfBestModels[i][1]:
			sfBestModels[i][1] = grad2
			sfBestModels[i][0] = meanModel2.copy()

def lrTest(eps,sfMask,nsf=50,na=2):

	bar = Bar('Likelihood ratio tests', max=np.count_nonzero(sfMask==0))
	super_policy = GaussianPolicy(nStateFeatures=nsf,actionDim=na)

	params = super_policy.estimate_params(eps,setToZero=None)
	ll = super_policy.getLogLikelihood(eps,params)

	ll_h0 = np.zeros(shape=(nsf),dtype=np.float32)
	for param in range(nsf):
		if sfMask[param]==False:
			params_h0 = super_policy.estimate_params(eps,setToZero=param)
			ll_h0[param] = super_policy.getLogLikelihood(eps,params_h0)
			bar.next()
	
	bar.finish()

	#print(ll)
	#print(ll_h0)
	lr_lambda = -2*(ll_h0 - ll)

	x = chi2.ppf(0.99,2)
	for param in range(nsf):
		if lr_lambda[param] > x:
			sfMask[param] = True
	
	return lr_lambda

def saveStateImage(filename,mean,var,sfMask):
	W = H = 300
	S = W/5
	offset = S/2

	pf = 7
	gf = 10

	image = Image.new("RGB", (W, H), (255,255,255))
	draw = ImageDraw.Draw(image)

	center_pos = np.array([mean[0]+2.5,5-(mean[1]+2.5)],dtype=np.float32)/5*W
	center_goal = np.array([mean[2]+2.5,5-(mean[3]+2.5)],dtype=np.float32)/5*W
	dist_pos = np.array([var[0],var[1]],dtype=np.float32)*S*1.5
	dist_goal = np.array([var[2],var[3]],dtype=np.float32)*S*2
	draw.ellipse((center_pos[0]-dist_pos[0],center_pos[1]-dist_pos[1],center_pos[0]+dist_pos[0],center_pos[1]+dist_pos[1]),fill=(200,200,255))
	draw.ellipse((center_goal[0]-dist_goal[0],center_goal[1]-dist_goal[1],center_goal[0]+dist_goal[0],center_goal[1]+dist_goal[1]),fill=(255,200,200))

	#draw.line((W,0,W,H),fill=(0,0,0),width=2)
	for i in range(5):
		for j in range(5):
			draw.rectangle((offset+j*S-1, offset+(4-i)*S-1, offset+j*S+1, offset+(4-i)*S+1), fill=(0,0,0), outline=(0,0,0))
			draw.rectangle((W+offset+j*S-1, offset+(4-i)*S-1, W+offset+j*S+1, offset+(4-i)*S+1), fill=(0,0,0), outline=(0,0,0))
			posindex = j+i*5
			goalindex = 25+j+i*5
			if sfMask[posindex]:
				draw.ellipse((offset+j*S-pf, offset+(4-i)*S-pf, offset+j*S+pf, offset+(4-i)*S+pf),outline=(0,0,255))
			if sfMask[goalindex]:
				draw.ellipse((offset+j*S-gf, offset+(4-i)*S-gf, offset+j*S+gf, offset+(4-i)*S+gf),outline=(255,0,0))

	image.save(filename)

def saveTrajectoryImage(filename,history_pos,history_goal):
	W = H = 300
	S = W/5
	offset = S/2

	pf = 7
	gf = 10

	image = Image.new("RGB", (W, H), (255,255,255))
	draw = ImageDraw.Draw(image)

	for i in range(5):
		for j in range(5):
			draw.rectangle((offset+j*S-1, offset+(4-i)*S-1, offset+j*S+1, offset+(4-i)*S+1), fill=(0,0,0), outline=(0,0,0))
			draw.rectangle((W+offset+j*S-1, offset+(4-i)*S-1, W+offset+j*S+1, offset+(4-i)*S+1), fill=(0,0,0), outline=(0,0,0))

	for i in range(len(history_pos)-1):
		center_pos_a = np.array([history_pos[i][0]+2.5,5-(history_pos[i][1]+2.5)],dtype=np.float32)/5*W
		center_pos_b = np.array([history_pos[i+1][0]+2.5,5-(history_pos[i+1][1]+2.5)],dtype=np.float32)/5*W
		center_goal_a = np.array([history_goal[i][0]+2.5,5-(history_goal[i][1]+2.5)],dtype=np.float32)/5*W
		center_goal_b = np.array([history_goal[i+1][0]+2.5,5-(history_goal[i+1][1]+2.5)],dtype=np.float32)/5*W
		draw.line((center_pos_a[0],center_pos_a[1],center_pos_b[0],center_pos_b[1]),fill=(0,0,255),width=2)
		draw.line((center_goal_a[0],center_goal_a[1],center_goal_b[0],center_goal_b[1]),fill=(255,0,0),width=2)

	image.save(filename)

def clearn_nn(mdp, policy, steps, nEpisodes, learningRate=0.1, plotGradient=False, printInfo=False):

	if steps<=0 or steps is None:
		plotGradient = False
	
	if plotGradient:
		xs = []
		mys = []
		ys = []
		plt.plot(xs,ys)
		plt.plot(xs,mys)
	
	optimizer = AdamOptimizer(policy.nParams, learning_rate=learningRate, beta1=0.9, beta2=0.99)

	avg = 0.95
	avg_mean_length = 0
	avg_max_gradient = 0
	avg_mean_gradient = 0
	mt = avg

	for step in range(steps):

		eps = collect_cgridworld_episodes(mdp,policy,nEpisodes,mdp.horizon,render=False)
		gradient = policy.optimize_gradient(eps,optimizer)

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
			print("Step: "+str(step))
			print("Mean length: "+str(np.round(mean_length,3)))
			print("Mean gradient: "+str(np.round(mean_gradient,5)))
			print("Maximum gradient: "+str(np.round(max_gradient,5)))
			print("Mean rewards: ",str(np.round(mean_reward,5)))

		if plotGradient:
			xs.append(step)
			ys.append(mean_reward)
			#mys.append(mean_gradient)
			plt.gca().lines[0].set_xdata(xs)
			plt.gca().lines[0].set_ydata(ys)
			#plt.gca().lines[1].set_xdata(xs)
			#plt.gca().lines[1].set_ydata(mys)
			plt.gca().relim()
			plt.gca().autoscale_view()
			#plt.yscale("log")
			plt.pause(0.000001)
		
		#policy.print_params()
		
		if step%25==0 and step>0:
			collect_cgridworld_episode(mdp,policy,25,render=True)