import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
from util.optimizer import *
from scipy.stats import chi2


def build_state_features(state,sfmask=None):
	x = np.array(state)
	if sfmask is not None:
		mask = np.array(sfmask, dtype=bool)
		x[mask==0] = 0
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


def learn(learner, steps, nEpisodes, initParams=None, sfmask=None, learningRate=0.1, plotGradient=False, printInfo=False):
	
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
	if initParams is not None:
		learner.policy.set_params(initParams)
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
		#print("Average mean length: "+str(np.round(avg_mean_length_t,3)))
		max_gradient = np.max(np.abs(gradient))
		avg_max_gradient = avg_max_gradient*avg+max_gradient*(1-avg)
		#print("Avg maximum gradient: "+str(np.round(avg_max_gradient_t,5)))
		mean_gradient = np.mean(np.abs(gradient))
		avg_mean_gradient = avg_mean_gradient*avg+mean_gradient*(1-avg)
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
			if learner.mdp.renderFlag:
				collect_car_episode(learner.mdp,learner.policy,learner.mdp.horizon,sfmask=sfmask,render=True)


def getModelGradient(superLearner, eps, Neps, sfTarget, model_w_new, model_w):
	
	N = Neps
	Tmax = max(eps["len"][:N])
	n_policy_params = superLearner.policy.nParams

	mdp = superLearner.mdp
	policy = superLearner.policy
	gamma = superLearner.gamma
	sf = eps["s"][:N]
	a = eps["a"][:N]
	r = eps["r"][:N]

	dr = np.zeros(shape=(N,Tmax),dtype=np.float32)

	pgrad = np.zeros(shape=(n_policy_params),dtype=np.float32)
	mgradpgrad = np.zeros(shape=(n_policy_params),dtype=np.float32)

	d2 = np.float32(0)
	mgrad_d2 = np.float32(0)
	is_ratio_d2_2nd = np.ones(shape=(N,Tmax),dtype=np.float32)
	model_log_grads_d2_2nd = np.zeros(shape=(N,Tmax),dtype=np.float32)

	for n in range(N):

		T = eps["len"][n]
		model_log_grad_t = np.float32(0)
		policy_log_grad_t = np.zeros(shape=(n_policy_params),dtype=np.float32)
		is_ratio_t = np.float32(1)

		for t in range(T):
			dr[n,t] = (gamma**t) * r[n,t]

			policy_log_grad_t += policy.compute_log_gradient(sf[n,t],a[n,t])

			if t<T-1:
				model_log_grad_t += mdp.grad_log_p_model(sf[n,t+1][0:7],sf[n,t],a[n,t],model_w_new)
				model_log_grads_d2_2nd[n,t] = np.copy(model_log_grad_t)

				is_ratio_t *= mdp.p_model(sf[n,t+1][0:7],sf[n,t],a[n,t],model_w_new)/mdp.p_model(sf[n,t+1][0:7],sf[n,t],a[n,t],model_w)
				is_ratio_d2_2nd[n,t] = np.copy(is_ratio_t)		# 2nd d2 estimator
				#d2 += (gamma**t)*(gamma**t+2*gamma**(t+1)-2*gamma**T)*(is_ratio_t**2)
				#mgrad_d2 += 2*(gamma**t)*(gamma**t+2*gamma**(t+1)-2*gamma**T)*(is_ratio_t**2)*model_log_grad_t

			pgrad_t = dr[n,t]*policy_log_grad_t * (is_ratio_t if t<T-1 else 1)
			pgrad += pgrad_t
			mgradpgrad += pgrad_t*(model_log_grad_t if t<T-1 else 1)

	pgrad /= N
	mgradpgrad /= N
	#d2 /= N*(1-gamma)			# 1st d2 estimator
	#mgrad_d2 /= N*(1-gamma)	# 1st d2 estimator

	# 2nd d2 estimator
	for t in range(Tmax):
		d2_t = np.float32(0)
		mgrad_d2_t = np.float32(0)
		for n in range(N):
			T = eps["len"][n]
			if(t<T-1):
				d2_t += (is_ratio_d2_2nd[n,t]-1)**2
				mgrad_d2_t += 2*(is_ratio_d2_2nd[n,t]-1)*is_ratio_d2_2nd[n,t]*model_log_grads_d2_2nd[n,t]
		d2_t /= N
		d2_t += 1
		d2 += (gamma**t)*(gamma**t+2*(gamma**(t+1))-2*(gamma**Tmax))*d2_t
		mgrad_d2 /= N
		mgrad_d2 += (gamma**t)*(gamma**t+2*(gamma**(t+1))-2*(gamma**Tmax))*mgrad_d2_t
	d2 /= (1-gamma)
	mgrad_d2 /= (1-gamma)
	
	wnum = policy.params["w1"].shape[1]
	model_term = np.dot(pgrad[sfTarget*wnum:(sfTarget+1)*wnum],mgradpgrad[sfTarget*wnum:(sfTarget+1)*wnum])

	lambda_param = 1/4
	d2_term = lambda_param/(2*np.sqrt(N))*mgrad_d2/np.sqrt(d2)

	print("\nd2        =",d2,flush=True)
	print("model_term  =",model_term,flush=True)
	print("d2_term     =",d2_term,flush=True)

	return model_term - d2_term


def lrTest(eps,policyInstance,sfMask,nsf,na,lr=0.01,batchSize=100,epsilon=0.0001,maxSteps=10000,numResets=1):

	bar = Bar('Likelihood ratio tests', max=numResets*(np.count_nonzero(sfMask==0)+1))

	ll_tot_list = []
	for _ in range(numResets):
		policyInstance.estimate_params(eps,lr,nullFeatures=None,batchSize=batchSize,epsilon=epsilon,maxSteps=maxSteps)
		ll_tot_list.append(policyInstance.getLogLikelihood(eps))
		bar.next()
	ll_tot = np.max(ll_tot_list)

	ll_h0 = np.zeros(shape=(nsf),dtype=np.float32)
	#ll_tot = np.zeros(shape=(nsf),dtype=np.float32)
	for feature in range(nsf):
		if not sfMask[feature]:
			ll0 = []
			#ll = []
			for _ in range(numResets):
				params0 = policyInstance.estimate_params(eps,lr,nullFeatures=[feature],batchSize=batchSize,epsilon=epsilon,maxSteps=maxSteps,printInfo=False)
				ll0.append(policyInstance.getLogLikelihood(eps))
				#policyInstance.estimate_params(eps,lr,params0=params0,nullFeatures=None,batchSize=batchSize,epsilon=epsilon,maxSteps=maxSteps,printInfo=False)
				#ll.append(policyInstance.getLogLikelihood(eps))
			
			ll_h0[feature] = np.max(ll0)
			#ll_tot[feature] = np.max(ll)
			bar.next()

	bar.finish()

	print("[DEBUG] Log likelihood without i-th feature:",ll_h0)
	print("[DEBUG] Log likelihood with every feature:  ",ll_tot)
	lr_lambda = -2*(ll_h0 - ll_tot)
	print("[DEBUG] LR lambda: ",lr_lambda)

	x = chi2.ppf(0.99,policyInstance.nHiddenNeurons)
	print("[DEBUG] chi2 statistic: ",x)
	for param in range(nsf):
		if lr_lambda[param] > x:
			sfMask[param] = True

	return lr_lambda

def lrCombTest(eps,policyInstance,nsf,na,lr=0.01,batchSize=100,epsilon=0.0001,maxSteps=10000):

	bar = Bar('Likelihood ratio tests', max=2**nsf)

	ll = np.zeros(shape=(2**nsf),dtype=np.float32)
	for f in range(2**nsf-1):
		if f>0:
			mask = np.base_repr(f,padding=nsf)
			mask = [int(i) for i in mask[-nsf:]]
			null_features = np.array(np.where(mask))
		else:
			null_features = None
		params = policyInstance.estimate_params(eps,lr,nullFeatures=null_features,batchSize=batchSize,epsilon=epsilon,maxSteps=maxSteps,printInfo=True)
		ll[f] = policyInstance.getLogLikelihood(eps)
		bar.next()

	bar.finish()

	print(ll)

	ll_tot = ll[0]
	ll_part = ll[1:2**nsf-1]
	lr_lambda = -2*(ll_part - ll_tot)

	print("[DEBUG] ll_tot:    ",ll_tot)
	print("[DEBUG] ll_part:   ",ll_part)
	print("[DEBUG] LR lambda: ",lr_lambda)

	sfCombMask = np.zeros(shape=2**nsf-2,dtype=np.bool)
	sfLinMask = np.zeros(shape=nsf,dtype=np.bool)

	subsets = []
	#subsets_params = []

	for i in range(2**nsf-2):
		mask = np.base_repr(i+1,padding=nsf)
		mask = np.array([int(j) for j in mask[-nsf:]],dtype=np.bool)

		nullFeaturesWeight = np.sum(mask)
		subsetWeight = nsf - nullFeaturesWeight

		x = chi2.ppf(0.99,nullFeaturesWeight*policyInstance.nHiddenNeurons)
		if lr_lambda[i]>x:
			sfCombMask[i] = True
			if nullFeaturesWeight==1:
				index = np.where(mask==True)
				sfLinMask[index] = True
		else:
			ok = True
			for j in range(2**nsf-2):
				mask_j = np.base_repr(j+1,padding=nsf)
				mask_j = np.array([int(k) for k in mask_j[-nsf:]],dtype=np.bool)
				nullFeaturesWeight_j = np.sum(mask_j)
				subsetWeight_j = nsf - nullFeaturesWeight_j
				if subsetWeight_j==subsetWeight-1 and np.sum(mask_j[mask==True])==nullFeaturesWeight:
					y = chi2.ppf(0.99,nullFeaturesWeight*policyInstance.nHiddenNeurons)
					if lr_lambda[j]<y:
						ok = False
			if ok:
				subsets.append(1-mask)
				#subsets_params.append()

	return [subsets,sfLinMask]