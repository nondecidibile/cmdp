import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
from util.optimizer import *
from scipy.stats import chi2
from tqdm import tqdm
from util.policy_gaussian import *


def build_state_features(state, sfmask=None):
	x, f = state
	x = max(0, x)
	features = np.array([1, x, f, np.sqrt(x), np.sqrt(f), np.sqrt(x * f)])
	if sfmask is not None:
		return features[sfmask]
	return features


def collect_minigolf_episode(mdp,policy,horizon,sfmask=None,exportAllStateFeatures=False):

	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or sfmask is None) else len(sfmask)
	original_states = np.zeros(shape=(horizon,2),dtype=np.float32)
	states = np.zeros(shape=(horizon,nsf),dtype=np.float32)
	actions = np.zeros(shape=(horizon,1),dtype=np.float32)
	rewards = np.zeros(shape=horizon,dtype=np.float32)

	state = mdp.reset()
	length = 0

	for i in range(horizon):

		length += 1

		action = policy.draw_action(build_state_features(state,sfmask))
		newstate, reward, done, _ = mdp.step(action)

		original_states[i] = state
		states[i] = build_state_features(state,(None if exportAllStateFeatures else sfmask))
		actions[i] = action
		rewards[i] = reward
		
		state = newstate

		if done:
			break
	
	episode_data = {"state": original_states, "s": states,"a": actions,"r": rewards}
	return [episode_data,length]


def collect_minigolf_episodes(mdp,policy,num_episodes,horizon,sfmask=None,showProgress=False,exportAllStateFeatures=False):
	
	nsf = policy.nStateFeatures if (exportAllStateFeatures is False or sfmask is None) else len(sfmask)
	data_states = np.zeros(shape=(num_episodes,horizon,2),dtype=np.float32)
	data_s = np.zeros(shape=(num_episodes,horizon,nsf),dtype=np.float32)
	data_a = np.zeros(shape=(num_episodes,horizon,1),dtype=np.float32)
	data_r = np.zeros(shape=(num_episodes,horizon),dtype=np.float32)
	data_len = np.zeros(shape=num_episodes, dtype=np.int32)
	data = {"state":data_states, "s": data_s, "a": data_a, "r": data_r, "len": data_len}

	if showProgress:
		bar = Bar('Collecting episodes', max=num_episodes)
	
	for i in range(num_episodes):
		episode_data, length = collect_minigolf_episode(mdp,policy,horizon,sfmask,exportAllStateFeatures)
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

	optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=learningRate, beta1=0.9, beta2=0.99)

	avg = 0.95
	avg_mean_length = 0
	avg_max_gradient = 0
	avg_mean_gradient = 0
	mt = avg

	#print("step, mean_reward, max_gradient, mean_update")

	best = -np.inf

	for step in (tqdm(range(steps)) if printInfo else range(steps)):
		#print(step)

		eps = collect_minigolf_episodes(learner.mdp,learner.policy,nEpisodes,learner.mdp.horizon,sfmask)
		#print(eps['r'])

		gradient = learner.estimate_gradient(eps)
		update_step = optimizer.step(gradient)

		print(repr(learner.policy.params))

		learner.policy.params += update_step


		mean_length = np.mean(eps["len"])
		avg_mean_length = avg_mean_length*avg+mean_length*(1-avg)
		#print("Average mean length: "+str(np.round(avg_mean_length_t,3)))
		max_gradient = np.max(np.abs(gradient))
		avg_max_gradient = avg_max_gradient*avg+max_gradient*(1-avg)
		#print("Avg maximum gradient: "+str(np.round(avg_max_gradient_t,5)))
		mean_gradient = np.mean(np.abs(gradient))
		avg_mean_gradient = avg_mean_gradient*avg+mean_gradient*(1-avg)
		#print("Avg mean gradient: "+str(np.round(avg_mean_gradient_t,5))+"\n")
		mean_reward = np.mean(np.sum(eps["r"], axis=1))
		best = max(best, mean_reward)
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
			if False:
				ep = collect_minigolf_episode(learner.mdp,learner.policy,learner.mdp.horizon,sfmask=sfmask)
				print(ep[0]["state"][:ep[1]],ep[0]["a"][:ep[1]],ep[0]["r"][:ep[1]])

	return best


def getModelGradient(superLearner, eps, Neps, sfTarget, model_w_new, model_w):
	
	N = Neps
	n_policy_params = superLearner.policy.nParams

	mdp = superLearner.mdp
	policy = superLearner.policy
	gamma = superLearner.gamma
	states = eps["state"][:N]
	sf = eps["s"][:N]
	a = eps["a"][:N]
	r = eps["r"][:N]

	pgrad = np.zeros(shape=(n_policy_params),dtype=np.float32)
	mgradpgrad = np.zeros(shape=(n_policy_params),dtype=np.float32)

	d2 = 0
	mgrad_d2 = 0

	for n in range(N):

		T = eps["len"][n]

		p_model = mdp.p_model(states[n][1:T],a[0][0:T-1],states[n][0:T-1],model_w)
		p_model_new = mdp.p_model(states[n][1:T],a[0][0:T-1],states[n][0:T-1],model_w_new)
		is_ratios = np.insert(np.cumprod(p_model_new / (p_model+1e-09)),0,1)
		if T==1:
			is_ratios = np.array([1],dtype=np.float32)

		grad_log_p_model = mdp.grad_log_p_model(states[n][1:T],a[0][0:T-1],states[n][0:T-1],model_w_new)
		model_log_grads = np.insert(np.cumsum(grad_log_p_model),0,0)

		steps = np.arange(T)
		dr = r[n][0:T]*(gamma**steps)

		policy_log_grad_terms = np.squeeze(policy.compute_log_gradient(sf[n][0:T],a[n][0:T]))
		policy_log_grads = np.reshape(np.cumsum(policy_log_grad_terms,axis=0),(-1,6))

		mgrad_is_ratios = is_ratios*model_log_grads
		if T==1:
			mgrad_is_ratios = np.array([0],dtype=np.float32)

		pgrad += np.dot(dr*is_ratios,policy_log_grads)
		mgradpgrad += np.dot(dr*is_ratios*mgrad_is_ratios,policy_log_grads)

		gamma_terms = (gamma**steps) * ((gamma**steps)+2*(gamma**(steps+1))+gamma**T)
		is_d2_terms = is_ratios**2
		d2 += np.sum(gamma_terms*is_d2_terms)
		mgrad_d2 += np.sum(2*gamma_terms*is_d2_terms*model_log_grads)
	
	pgrad /= N
	mgradpgrad /= N
	d2 /= N*(1-gamma)
	mgrad_d2 /= N*(1-gamma)

	model_term = np.dot(pgrad[sfTarget],mgradpgrad[sfTarget])

	lambda_param = 0.25
	d2_term = lambda_param/(2*np.sqrt(N))*mgrad_d2/np.sqrt(d2)

	return model_term - d2_term

def lrTest(eps,sfMask,nsf=6,na=1):

	ntest = np.count_nonzero(sfMask==0)
	bar = Bar('Likelihood ratio tests', max=ntest)
	super_policy = GaussianPolicy(nStateFeatures=nsf,actionDim=na)

	params = super_policy.estimate_params(eps,setToZero=None)
	ll = super_policy.getLogLikelihood(eps,params)

	ll_h0 = np.zeros(shape=(nsf),dtype=np.float32)
	for param in range(nsf):
		if sfMask[param]==False:
			params_h0 = super_policy.estimate_params(eps,setToZero=param)
			ll_h0[param] = super_policy.getLogLikelihood(eps,params_h0)
			bar.next()
		else:
			ll_h0[param] = ll # so that we have 0 in the already rejected features
	
	bar.finish()

	#print(ll)
	#print(ll_h0)
	lr_lambda = -2*(ll_h0 - ll)

	x = chi2.ppf(0.99,1)
	for param in range(nsf):
		if lr_lambda[param] > x:
			sfMask[param] = True
	
	return lr_lambda