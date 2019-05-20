from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
import os

basename = os.path.basename(__file__).split('.')[0]
start_time = datetime.datetime.now().strftime('%m%d%H%M%S')
print(basename, start_time)

N_BITS = 20

N_TIMES = 1
N_EP = 100000
N_STEP = N_BITS * 4
N_WARMUP_EP = 5

N_MEM = 3000000
N_BATCH = 32

SDIM = N_BITS * 2
ADIM = 1
N_ACTION = N_BITS

save_reward = np.zeros((N_TIMES, N_EP))

dqn = DQN(SDIM, N_ACTION, ADIM)
mem = ReplayMemory(SDIM, ADIM, N_MEM)

env = BitFlipEnv(N_BITS)

for t in range(N_TIMES):
    dqn.reset()
    mem.reset()
    for ep in range(N_EP):
        current, target = env.reset()
        state = np.hstack([current, target])
        for step in range(N_STEP):
            action = dqn.choose_action(state)
            state_next, reward, done, _ = env.step(action)
            
            current, target = state_next
            state_next = np.hstack([current, target])
            
            mem.store_transition(state, state_next, action, reward)
            
            if ep >= N_WARMUP_EP:
                dqn.learn(mem, N_BATCH)
            
            state = state_next
            
            if done:
                break
                
        print("T: {t}/{tt} E: {ep}/{ept} S: {step} R: {reward}".format(t=t, tt=N_TIMES, ep=ep, ept=N_EP, step=step, reward=reward / (step + 1)))
        save_reward[t, ep] = reward / (step + 1)
    
sio.savemat('logs/%s_%s.mat' % (basename, start_time), {'reward': save_reward})
end_time = datetime.datetime.now().strftime('%m%d%H%M%S')
print(basename, start_time, end_time)
    
plt.plot(save_reward.mean(0))
plt.show()
