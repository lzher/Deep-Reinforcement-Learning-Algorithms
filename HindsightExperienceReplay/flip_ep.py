from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
from collections import namedtuple

start_time = datetime.datetime.now().strftime('%m%d%H%M%S')
print(start_time)

N_BITS = 5

N_TIMES = 10
N_EP = 1000
N_STEP = 20
N_WARMUP_EP = 5

N_MEM = 30000
N_BATCH = 32

SDIM = N_BITS * 2
ADIM = 1
N_ACTION = N_BITS

save_reward = np.zeros((N_TIMES, N_EP))
save_true_reward = np.zeros((N_TIMES, N_EP))

EpMemory = namedtuple('EpMemory', ['state', 'state_next', 'action', 'reward'])

dqn = DQN(SDIM, N_ACTION, ADIM)
mem = ReplayMemory(SDIM, ADIM, N_MEM)

env = BitFlipEnv(N_BITS)

for t in range(N_TIMES):
    dqn.reset()
    mem.reset()
    for ep in range(N_EP):
        current, target = env.reset()
        state = np.hstack([current, target])
        ep_mem = []
        for step in range(N_STEP):
            action = dqn.choose_action(state)
            state_next, reward, done, _ = env.step(action)
            
            current, target = state_next
            state_next = np.hstack([current, target])
            
            ep_mem.append(EpMemory(state=state, state_next=state_next, action=action, reward=reward))
            
            if ep >= N_WARMUP_EP:
                dqn.learn(mem, N_BATCH)
            
            state = state_next
            
            if done:
                break
                
        ep_reward = 0
        if done:
            for e in ep_mem:
                reward = e.reward + 1
                mem.store_transition(e.state, e.state_next, e.action, reward)
                ep_reward += reward
        else:
            for e in ep_mem:
                reward = e.reward - 1
                mem.store_transition(e.state, e.state_next, e.action, reward)
                ep_reward += reward
                
        print("T: {t}/{tt} E: {ep} S: {step} R: {reward}".format(t=t, tt=N_TIMES, ep=ep, step=step, reward=ep_reward / (step + 1)))
        save_reward[t, ep] = int(done) / (step + 1)
        save_true_reward[t, ep] = ep_reward / (step + 1)
    
sio.savemat('logs/flip_ep_%s.mat' % (start_time), {'reward': save_reward, 'true_reward': save_true_reward})
end_time = datetime.datetime.now().strftime('%m%d%H%M%S')
print(start_time, end_time)
    
plt.plot(save_reward.mean(0))
plt.show()
