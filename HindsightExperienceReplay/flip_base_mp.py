from dqn_base import DQN, ReplayMemory
from bitflipenv import BitFlipEnv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
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
    dqn = DQN(SDIM, N_ACTION, ADIM)
    mem = ReplayMemory(SDIM, ADIM, N_MEM)

    env = BitFlipEnv(N_BITS)
    
    save_reward = np.zeros((N_TIMES, N_EP))

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
                    
            print("P: {p} T: {t}/{tt} E: {ep} S: {step} R: {reward}".format(p=prci, t=t, tt=N_TIMES, ep=ep, step=step, reward=reward / (step + 1)))
            save_reward[t, ep] = reward / (step + 1)
    ret[prci] = save_reward

if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%m%d%H%M%S')
    print(start_time)
    
    mp.set_start_method('spawn')
    ret = mp.Manager().dict()
    
    save_reward = np.zeros((N_PROCESSES*N_TIMES, N_EP))
    
    prcs = [mp.Process(target=train_job, args=(i,ret)) for i in range(N_PROCESSES)]
    for prc in prcs:
        prc.start()
    for prc in prcs:
        prc.join()
    
    for prci in ret.keys():
        save_reward[prci*N_TIMES:(prci+1)*N_TIMES, :] = ret[prci]
    
    sio.savemat('logs/flip_base_%s.mat' % (start_time), {'reward': save_reward})
    end_time = datetime.datetime.now().strftime('%m%d%H%M%S')
    print(start_time, end_time)
        
    plt.plot(save_reward.mean(0))
    plt.show()
