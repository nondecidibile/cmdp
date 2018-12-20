import numpy as np
import time
from gym.envs.toy_text import taxi, gridworld

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


def build_gridworld_features(mdp,observation):

	lst = list(mdp.decode(observation))
	row, col, row_g, col_g = lst[0], lst[1], lst[2], lst[3]

	state_features = []

	# taxi position
	state_features += onehot_encode(row,4)
	state_features += onehot_encode(col,4)
	# dst position
	state_features += onehot_encode(row_g,4)
	state_features += onehot_encode(col_g,4)

	return state_features


def collect_gridworld_episode(mdp,policy,horizon,render=False):

	states = []
	actions = []
	rewards = []

	state = mdp.reset()
	if render:
		mdp.render()

	length = 0

	for i in range(horizon):

		length += 1

		state_features = build_gridworld_features(mdp,state)
		action = policy.draw_action(state_features)

		newstate, reward, done, _ = mdp.step(action)
		states.append(state_features)
		actions.append(action)
		if done:
			rewards.append(0)
		else:
			rewards.append(reward)

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


def collect_gridworld_episodes(mdp,policy,num_episodes,horizon,render=False):
	
	data = []

	mean_length = 0

	for i in range(num_episodes):
		episode_data, length = collect_gridworld_episode(mdp,policy,horizon,render)
		data.append(episode_data)
		mean_length += length
	
	mean_length /= num_episodes

	return [data,mean_length]
