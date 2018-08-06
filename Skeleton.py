overview

def rollout(controller):
’’’ env, rnn, vae are ’’’
’’’ global variables ’’’

obs = env.reset()
h = rnn.initial_state()
done = False
cumulative_reward = 0

while not done:
	z = vae.encode(obs)
	a = controller.action([z, h])
	obs, reward, done = env.step(a)
	cumulative_reward += reward
	h = rnn.forward([a, z, h])
return cumulative_reward


#abstract

es = cma(w0,sgma0, popsize)
f = rollout(controller)

