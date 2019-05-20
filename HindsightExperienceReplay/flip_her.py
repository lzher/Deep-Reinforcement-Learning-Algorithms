from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
from collections import namedtuple
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
        for e in ep_mem:
            reward = e.reward
            mem.store_transition(e.state, e.state_next, e.action, reward)
            ep_reward += reward
        if not done:
            fake_target = ep_mem[-1].state_next[:N_BITS]
            for i in range(len(ep_mem)):
                e = ep_mem[i]
                state = e.state.copy()
                state_next = e.state_next.copy()
                state[N_BITS:] = fake_target
                state_next[N_BITS:] = fake_target
                if i == step:
                    reward = e.reward + 1
                else:
                    reward = e.reward
                mem.store_transition(state, state_next, e.action, reward)
                
        print("T: {t}/{tt} E: {ep}/{ept} S: {step} R: {reward}".format(t=t, tt=N_TIMES, ep=ep, ept=N_EP, step=step, reward=ep_reward / (step + 1)))
        save_reward[t, ep] = int(done) / (step + 1)
        save_true_reward[t, ep] = ep_reward / (step + 1)
    
sio.savemat('logs/%s_%s.mat' % (basename, start_time), {'reward': save_reward})
end_time = datetime.datetime.now().strftime('%m%d%H%M%S')
print(basename, start_time, end_time)
    
plt.plot(save_reward.mean(0))
plt.show()
