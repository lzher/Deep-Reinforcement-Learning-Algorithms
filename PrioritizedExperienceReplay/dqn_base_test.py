from dqn_base import DQN, ReplayMemory
import gym
import datetime
import os
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Pendulum-v0').unwrapped

N_TIMES = 20
N_EP = 300
N_STEP = 200

N_WARMUP_EP = 20
N_REPLACE_TARGET = 200

N_MEM = 30000
N_BATCH = 32

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high[0]
a_val = 20
actions = np.linspace(-a_bound, a_bound, a_val)

now = datetime.datetime.now()
LOG_PATH = 'logs/R_' + os.path.basename(__file__) + now.strftime('_%Y_%m_%d_%H_%M_%S')
os.mkdir(LOG_PATH)
LOG_FILE = LOG_PATH + '/train.log'
log_fp = open(LOG_FILE, 'w')
print(LOG_FILE)

save_rewards = np.zeros((N_TIMES, N_EP))

dqn = DQN(s_dim, a_val, a_dim)
mem = ReplayMemory(s_dim, a_dim, N_MEM)

for t in range(N_TIMES):
    print("Time: %d" % t)
    dqn.reset()
    mem.reset()
    total_step = 0
    train_step = 0
    for ep in tqdm(range(N_EP)):
        ep_reward = 0
        step = 0
        state = env.reset()
        for step in range(N_STEP):
            # env.render()
            action = dqn.choose_action(state)
            act = actions[action]
            state_next, reward, done, _ = env.step([act])
            
            mem.store_transition(state, state_next, action, reward)
            
            if ep > N_WARMUP_EP:
                dqn.learn(mem, N_BATCH, replacement=(train_step % N_REPLACE_TARGET == 0))
                train_step += 1
                
            state = state_next
            ep_reward += reward
            total_step += 1
            
        report = "E: {ep} T: {ts} R: {reward}\n".format(
            ep=ep,
            step=step,
            reward=ep_reward,
            ts=total_step)
        log_fp.write(report)
        log_fp.flush()
        # print(report)
        save_rewards[t, ep] = reward

step_fn = LOG_PATH + '/rewards'
sio.savemat(step_fn, {'rewards': save_rewards})

plt.plot(save_rewards.mean(0))
plt.xlabel('Time slot')
plt.ylabel('Rewards')
plt.show()


