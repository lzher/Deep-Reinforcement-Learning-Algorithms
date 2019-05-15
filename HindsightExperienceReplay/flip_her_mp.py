from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
from collections import namedtuple
import torch.multiprocessing as mp


N_BITS = 5

N_PROCESSES = 5
N_TIMES = 10
N_EP = 1000
N_STEP = 20
N_WARMUP_EP = 5

N_MEM = 30000
N_BATCH = 32

SDIM = N_BITS * 2
ADIM = 1
N_ACTION = N_BITS

def train_job(prci, ret):
    save_reward = np.zeros((N_TIMES, N_EP))

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
                mem.store_transition(e.state, e.state_next, e.action, e.reward)
                ep_reward += e.reward
            if not done:
                fake_target = ep_mem[-1].state_next[:N_BITS]
                for i in range(len(ep_mem)):
                    e = ep_mem[i]
                    state = e.state.copy()
                    state_next = e.state_next.copy()
                    state[N_BITS:] = fake_target
                    state_next[N_BITS:] = fake_target
                    reward = e.reward
                    if(i == step):
                        reward += 1
                    mem.store_transition(state, state_next, e.action, reward)
                    
            print("P: {p} T: {t}/{tt} E: {ep} S: {step} R: {reward}".format(p=prci, t=t, tt=N_TIMES, ep=ep, step=step, reward=ep_reward / (step + 1)))
            save_reward[t, ep] = int(done) / (step + 1)
    ret[prci] = save_reward
    
if __name__ == '__main__':
    
    start_time = datetime.datetime.now().strftime('%m%d%H%M%S')
    print(start_time)
    
    mp.set_start_method('spawn')
    ret = mp.Manager().dict()
    
    save_reward = np.zeros((N_TIMES, N_EP))
    
    prcs = [mp.Process(target=train_job, args=(i,ret)) for i in range(N_PROCESSES)]
    for prc in prcs:
        prc.start()
    for prc in prcs:
        prc.join()
    
    for prci in ret.keys():
        save_reward[prci*N_TIMES:(prci+1)*N_TIMES, :] = ret[prci]
        
    sio.savemat('logs/flip_her_%s.mat' % (start_time), {'reward': save_reward})
    end_time = datetime.datetime.now().strftime('%m%d%H%M%S')
    print(start_time, end_time)
        
    plt.plot(save_reward.mean(0))
    plt.show()
