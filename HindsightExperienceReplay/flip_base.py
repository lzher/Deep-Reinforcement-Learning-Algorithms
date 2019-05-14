from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt

N_BITS = 5
N_EP = 3000
N_STEP = 20
N_WARMUP_EP = 5

N_MEM = 30000
N_BATCH = 32

SDIM = N_BITS * 2
ADIM = 1
N_ACTION = N_BITS

save_reward = np.zeros((N_EP))

dqn = DQN(SDIM, N_ACTION, ADIM)
mem = ReplayMemory(SDIM, ADIM, N_MEM)

env = BitFlipEnv(N_BITS)
current, target = env.reset()
state = np.hstack([current, target])

for ep in range(N_EP):
    for step in range(N_STEP):
        action = dqn.choose_action(state)
        state_next, reward, done, _ = env.step(action)
        
        current, target = state_next
        state_next = np.hstack([current, target])
        
        mem.store_transition(state, state_next, action, reward)
        
        if ep >= N_WARMUP_EP:
            dqn.learn(mem, N_BATCH)
        
        if done:
            break
            
    print("E: {ep} S: {step} R: {reward}".format(ep=ep, step=step, reward=reward))
    save_reward[ep] = reward
    
plt.plot(save_reward)
plt.show()
