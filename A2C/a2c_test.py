from a2c_base import A2C, ReplayMemory
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time

N_EP = 5000
N_STEP = 50
N_MEM = 3000000
N_WARMUP_EP = 50
N_BATCH = 32
N_TRAIN = 5

env = gym.make('Pendulum-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

a2c = A2C(N_S, N_A)
mem = ReplayMemory(N_S, N_A, N_MEM)
ep_mem = ReplayMemory(N_S, N_A, N_MEM)

save_rewards = np.zeros((N_EP))

TIME = time.strftime("%Y%m%d_%H%M%S")
print(TIME)

total_step = 0

for ep in range(N_EP):
    ep_reward = 0
    state = env.reset()
    ep_mem.reset()
    for step in range(N_STEP):
        action = a2c.choose_action(state).clip(-1, 1)
        env_action = action
        state_next, reward, done, _ = env.step(env_action)
        reward = (reward + 8.1) / 8.1
        
        ep_mem.store_transition(state, state_next, action, reward)
        
        ep_reward += reward
        
        done = (step == N_STEP-1)
        total_step += 1
        
        state = state_next
        
        if done or (total_step % N_TRAIN == 0):
            c_loss, a_loss = a2c.learn_episode(ep_mem, done)
    print("E: {ep}/{ept} R: {r}".format(ep=ep, ept=N_EP, r=ep_reward))
    save_rewards[ep] = ep_reward
    
print(TIME)
sio.savemat("logs/ac_test_{time}.mat".format(time=TIME), {"r": save_rewards})
plt.plot(save_rewards)
plt.show()

