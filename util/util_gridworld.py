import numpy as np
import time
from gym.envs.toy_text import taxi, gridworld
from util.policy_boltzmann import *
from util.optimizer import *
from progress.bar import Bar
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.stats import chi2


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


def build_gridworld_features(mdp,observation,transfMatrix=None,stateFeaturesMask=None):

	lst = list(mdp.decode(observation))
	row, col, row_g, col_g = lst[0], lst[1], lst[2], lst[3]

	state_features = []

	# taxi position
	state_features += onehot_encode(row,4)
	state_features += onehot_encode(col,4)
	# dst position
	state_features += onehot_encode(row_g,4)
	state_features += onehot_encode(col_g,4)
	# constant term
	state_features += [1]

	state_features = np.array(state_features)
	
	if transfMatrix is not None:
		state_features = np.dot(transfMatrix,state_features)

	if stateFeaturesMask is not None:
		mask = np.array(stateFeaturesMask, dtype=bool)
		state_features = state_features[mask]
	return state_features


def collect_gridworld_episode(mdp,policy,horizon,transfMatrix=None,stateFeaturesMask=None,exportAllStateFeatures=True,render=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)

	states = np.zeros(shape=(horizon,nsf),dtype=np.float32)
	actions = np.zeros(shape=horizon,dtype=np.int32)
	rewards = np.zeros(shape=horizon,dtype=np.int32)

	state = mdp.reset()
	initialState = np.array(list(mdp.decode(state)))
	if render:
		mdp.render()

	length = 0

	for i in range(horizon):

		length += 1

		state_features = build_gridworld_features(mdp,state,transfMatrix=transfMatrix,stateFeaturesMask=stateFeaturesMask)
		action = policy.draw_action(state_features)

		if exportAllStateFeatures is True:
			state_features = build_gridworld_features(mdp,state,transfMatrix=transfMatrix)

		newstate, reward, done, _ = mdp.step(action)

		states[i] = state_features
		actions[i] = action
		if done:
			rewards[i] = 0
		else:
			rewards[i] = -1

		if render:
			mdp.render()
			time.sleep(0.1)

		if done:
			break
		
		state = newstate
	
	episode_data = {"s": states, "a": actions, "r": rewards, "s0": initialState}
	return [episode_data,length]


def collect_gridworld_episodes(mdp,policy,num_episodes,horizon,transfMatrix=None,stateFeaturesMask=None,exportAllStateFeatures=True,render=False,showProgress=False):
	
	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or stateFeaturesMask is None) else len(stateFeaturesMask)

	data_s = np.zeros(shape=(num_episodes,horizon,nsf),dtype=np.float32)
	data_a = np.zeros(shape=(num_episodes,horizon),dtype=np.int32)
	data_r = np.zeros(shape=(num_episodes,horizon),dtype=np.int32)
	data_s0 = np.zeros(shape=(num_episodes,4),dtype=np.int32)
	data_len = np.zeros(shape=num_episodes, dtype=np.int32)
	data = {"s": data_s, "a": data_a, "r": data_r, "len": data_len, "s0": data_s0}

	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)

	for i in range(num_episodes):
		episode_data, length = collect_gridworld_episode(mdp,policy,horizon,transfMatrix,stateFeaturesMask,exportAllStateFeatures,render)
		data["s"][i] = episode_data["s"]
		data["a"][i] = episode_data["a"]
		data["r"][i] = episode_data["r"]
		data["s0"][i] = episode_data["s0"]
		data["len"][i] = length
		if showProgress:
			bar.next()

	if showProgress:
		bar.finish()	

	return data


def learn(learner, steps, nEpisodes, transfMatrix=None, sfmask=None, adamOptimizer=True,
		learningRate=0.1, loadFile=None, saveFile=None, autosave=False, plotGradient=False, printInfo=True):

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
	mt = avg

	if not printInfo:
		bar = Bar('Agent learning', max=steps)

	for step in range(steps):

		eps = collect_gridworld_episodes(learner.mdp,learner.policy,nEpisodes,learner.mdp.horizon,
				transfMatrix=transfMatrix,stateFeaturesMask=sfmask,exportAllStateFeatures=False)

		gradient = learner.estimate_gradient(eps)
		if adamOptimizer:
			update_step = optimizer.step(gradient)
		else:
			update_step = learningRate*gradient

		learner.policy.params += update_step

		mean_length = np.mean(eps["len"])
		avg_mean_length = avg_mean_length*avg+mean_length*(1-avg)
		avg_mean_length_t = avg_mean_length/(1-mt)
		max_gradient = np.max(np.abs(gradient))
		avg_max_gradient = avg_max_gradient*avg+max_gradient*(1-avg)
		avg_max_gradient_t = avg_max_gradient/(1-mt)
		mt = mt*avg
		if printInfo:
			print("Step:             "+str(step))
			print("Mean length:      "+str(np.round(mean_length,3)))
			print("Maximum gradient: "+str(np.round(max_gradient,5))+"\n")
		else:
			bar.next()

		if plotGradient:
			xs.append(step)
			ys.append(max_gradient)
			mys.append(avg_max_gradient_t)
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
	
	if not printInfo:
		bar.finish()
	
	'''
	if plotGradient:
		plt.show()
	'''


def find_params_ml(estLearner,eps,saveFile=None):

	print("Estimating policy parameters with Maximum Likelihood...\n")
	optimizer = AdamOptimizer(estLearner.policy.paramsShape,learning_rate=2.5)
	est_params = estLearner.policy.estimate_params(eps,optimizer,estLearner.policy.params,epsilon=0.01,minSteps=50,maxSteps=200)
		
	#_,est_ml = collect_gridworld_episodes(mdp,estLearner.policy,500,mdp.horizon)
	#print("Estimated avg mean length: ",est_ml,"\n")
	#p_abs_mean = np.mean(np.abs(estLearner.policy.params),axis=0)
	#print("Parameters absolute average between actions\n",p_abs_mean,"\n")

	if saveFile is not None:
		np.save(saveFile,est_params)


def d2divTerm(mdpP, mdpQ):
	posrows = np.zeros(625,dtype=np.int32)
	poscols = np.zeros(625,dtype=np.int32)
	goalrows = np.zeros(625,dtype=np.int32)
	goalcols = np.zeros(625,dtype=np.int32)
	dQxdw = np.zeros(shape=(625,20),dtype=np.float32)

	i = 0
	for a in range(5):
		for b in range(5):
			for c in range(5):
				for d in range(5):
					posrows[i] = a
					poscols[i] = b
					goalrows[i] = c
					goalcols[i] = d
					dQxdw[i] = mdpQ.dInitialProb_dw(a,b,c,d)
					i+=1

	probsP = mdpP.initialProb(posrows,poscols,goalrows,goalcols)
	probsQ = mdpQ.initialProb(posrows,poscols,goalrows,goalcols)
	d2div = np.sum((probsP**2)/probsQ)

	d_d2div_dw = np.dot(-((probsP/probsQ)**2),dQxdw)

	return d_d2div_dw/np.sqrt(d2div)


def getModelGradient(superLearner, eps, sfGradientMask, model, model2):

	sfGradientMaskTiled = np.tile(sfGradientMask,(4,1))
	sfGradientMaskLin = np.ravel(sfGradientMaskTiled)

	mdp = gridworld.GridworldEnv(model)
	mdp2 = gridworld.GridworldEnv(model2)

	# importance sampling
	initialStates = eps["s0"]
	initialProbs = mdp.initialProb(initialStates[:,0],initialStates[:,1],initialStates[:,2],initialStates[:,3])
	initialProbs2 = mdp2.initialProb(initialStates[:,0],initialStates[:,1],initialStates[:,2],initialStates[:,3])
	initialIS = initialProbs2 / initialProbs
	estimated_gradient2 = superLearner.estimate_gradient(eps,initialIS)

	# norm^2_2 of the gradient of J
	#x = np.linalg.norm(estimated_gradient2[sfGradientMaskTiled])**2

	grads_estimates = superLearner.estimate_gradient(eps,getEstimates=True)
	dj = np.ravel(estimated_gradient2)

	ddj = np.zeros(shape=(len(eps["len"]),model.size,len(sfGradientMaskLin)),dtype=np.float32)
	for n in range(len(eps["len"])):
		hx1 = mdp.dInitialProb_dw(initialStates[n,0],initialStates[n,1],initialStates[n,2],initialStates[n,3])
		kx1 = grads_estimates[n]
		ddj[n] = np.outer(hx1,kx1)
	ddj = (ddj.T*initialIS).T
	ddj = np.sum(ddj,axis=0)/len(eps["len"])

	dj = dj[sfGradientMaskLin]
	ddj = ddj[:,sfGradientMaskLin]
	model_gradient_J = np.matmul(ddj,dj)

	# ranyi divergence term
	lambda_param = 1/8*np.linalg.norm(estimated_gradient2,ord=np.inf)*np.sqrt(np.count_nonzero(sfGradientMaskLin)/0.95)
	#print("lambda =",lambda_param)
	#print("d2 = ",d2gaussians(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel)))
	#print("weights = ",np.mean(initialIS**2))

	model_gradient_div = lambda_param/2/np.sqrt(len(eps["len"])) * d2divTerm(mdp,mdp2)
	#model_gradient_div *= d_d2gaussians_dmuP(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel))
	#model_gradient_div /= np.sqrt(d2gaussians(meanModel2,np.diag(varModel2),meanModel,np.diag(varModel)))

	model_gradient = model_gradient_J - model_gradient_div
	#print("norm_2^2(grad J) =",x)
	#print("model gradient [J] =",model_gradient_J)
	#print("model gradient [div] =",model_gradient_div)
	#print("model gradient =",model_gradient)
	return np.reshape(model_gradient,newshape=(4,5))


def lrTest(eps,sfMask,lr=0.03,epsilon=0.001,maxSteps=1000):

	super_policy = BoltzmannPolicy(17,4)
	optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=lr)

	bar = Bar('Likelihood ratio tests', max=np.count_nonzero(sfMask==0))
	params = super_policy.estimate_params(eps,optimizer,setToZero=None,epsilon=0.001,minSteps=100,maxSteps=1000,printInfo=False)
	ll = super_policy.getLogLikelihood(eps,params)
	ll_h0 = np.zeros(shape=(16),dtype=np.float32)
	for param in range(16):
		if not sfMask[param]:
			optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=lr)
			params_h0 = super_policy.estimate_params(eps,optimizer,setToZero=param,epsilon=0.001,minSteps=100,maxSteps=1000,printInfo=False)
			ll_h0[param] = super_policy.getLogLikelihood(eps,params_h0)
			bar.next()
	bar.finish()
	
	#print(ll)
	#print(ll_h0)
	lr_lambda = -2*(ll_h0 - ll)

	x = chi2.ppf(0.99,4)
	for param in range(16):
		if lr_lambda[param] > x:
			sfMask[param] = True
	
	return lr_lambda


def saveStateImage(filename,mdp,sfMask):
	W = H = 200
	S = W/5

	image = Image.new("RGB", (W, H), (255,255,255))
	draw = ImageDraw.Draw(image)

	probs = mdp.getInitialProbs()

	for i in range(5):
		for j in range(5):
			fraction_pos = probs[0][i] * probs[1][j]
			fraction_goal = probs[2][i] * probs[3][j]
			color = fraction_pos*np.array([1,1,0]) + fraction_goal*np.array([0,1,1])
			color = 1-color#**0.5
			color = tuple(np.int32(255*color))
			draw.rectangle([S*j,4*S-S*i,S*j+S,5*S-S*i],fill=color,outline=(0,0,0))
	
	for i in range(4):
		if sfMask[4+i]:
			draw.line((i*S+S/2, S/10, i*S+S/2, H-S/10),fill=(0,0,255),width=1)
		if sfMask[i]:
			draw.line((S/10, i*S+S/2, W-S/10, i*S+S/2),fill=(0,0,255),width=1)
		if sfMask[12+i]:
			draw.line((i*S+S/2+3, S/10, i*S+S/2+3, H-S/10),fill=(255,0,0),width=1)
		if sfMask[8+i]:
			draw.line((S/10, i*S+S/2+3, W-S/10, i*S+S/2+3),fill=(255,0,0),width=1)

	image.save(filename)