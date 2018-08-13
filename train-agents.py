"""
Game is solved when agent consistently gets 900+ points. Track is random every episode.
"""

import numpy as np
import gym
import time, tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp

from train_VAE import load_vae

_z_size = 32
_h_size =256
_num_pred = 2
_num_as = 3
_num_param = _num_pred * _z_size + _num_pred

def normalize_obs(obs):
	return obs.astype('float32') / 255.

def get_weights_bias(params):
	weights = params[:_num_param - _num_pred]
	bias = params[-_num_pred:]
	weights = np.reshape(weights, [_z_size, _num_pred])
	return weights, bias

def vae_

def decide_a(sess, network, z, h, params):
	weights, bias = get_weights_bias(params)
	a = np.zeros(_num_as)

	#add rnn hidden to the equation
	pred = np.matmul(np.squeeze(z +h ), weights) + bias
	pred = np.tanh(pred)

	a[0] = pred[0]
	if pred[1] < 0:
		a[1] = np.abs(pred[1])
		a[2] = 0
	else:
		a[2] = pred[1]
		a[1] = 0

	return a

def rnn(z, a, h)
	return 
'''
def rollout(controller):
’’’ env, rnn, vae are ’’’
’’’ global variables ’’’

obs = env.reset()
h = rnn.initial_state()
done = False
cumulative_reward = 0

while not done:
	z = network.encoder(obs)
	a = decide_a(sess, network, obs, h, params)
	obs, reward, done = env.step(a)
	cumulative_reward += reward
	h = rnn.forward([a, z, h])
return cumulative_reward
'''

env = CarRacing()

def rollout(params, render=True, verbose=False):
	sess, network = load_vae()
	_num_trials = 12
	agent_reward = 0
	for trial in range(_num_trials):
		obs = env.reset()
		
		# Little hack to make the Car start at random positions in the race-track
		np.random.seed(int(str(time.time()*1000000)[10:13]))
		position = np.random.randint(len(env.track))
		env.car = Car(env.world, *env.track[position][1:4])
		
		#initalize
		cumulative_reward = 0.0
		steps = 0
		done = False
		h = np.zeros(256)
		W = np.random(256, 291)

		#rollout
		for steps in range(1000)
			if render:
				env.render()
			obs = normalize_obs(obs)
			z = sess.run(network.z, feed_dict={network.image: obs[None, :,  :,  :]})			
			a = decide_a(sess, network, z, h, params)
			# generate step reward
			obs, r, done, info = env.step(a)
			cumulative_reward += r

			# NB: done is not True after 1000 steps when using the hack above for
			# 	  random init of position
			if verbose and (steps % 200 == 0 or steps == 999):
				print("\na " + str(["{:+0.2f}".format(x) for x in a]))
				print("step {} cumulative_reward {:+0.2f}".format(steps, cumulative_reward))

			# add hidden

			h = np.matmul( W,  np.concatenate(a, z, h))

		# If reward is out of scale, clip it
		cumulative_reward = np.maximum(-100, cumulative_reward)
		agent_reward += cumulative_reward
	return - (agent_reward / _num_trials)

def train():
	es = cma.CMAEvolutionStrategy(_num_param * [0], 0.1, {'popsize': 16})
	rewards_through_gens = []
	generation = 1
	try:
		while not es.stop():
			solutions = es.ask()
			with mp.Pool(mp.cpu_count()) as p:
				rewards = list(tqdm.tqdm(p.imap(rollout, list(solutions)), total=len(solutions)))

			es.tell(solutions, rewards)

			rewards = np.array(rewards) *(-1.)
			print("\n**************")
			print("Generation: {}".format(generation))
			print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
			print("Avg reward: {:.3f}".format(np.mean(rewards)))
			print("**************\n")

			generation+=1
			rewards_through_gens.append(rewards)
			np.save('rewards', rewards_through_gens)

	except (KeyboardInterrupt, SystemExit):
		print("Manual Interrupt")
	except Exception as e:
		print("Exception: {}".format(e))
	return es

if __name__ == '__main__':
	es = train()
	np.save('best_params', es.best.get()[0])
	input("Press enter to rollout... ")
	RENDER = True
	score = rollout(es.best.get()[0], render=RENDER, verbose=True)
	print("Final Score: {}".format(-score))
