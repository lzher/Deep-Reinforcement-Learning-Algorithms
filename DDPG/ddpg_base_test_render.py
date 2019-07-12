from ddpg_base import DDPG, ReplayMemory
import gym
import datetime
import os
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Pendulum-v0').unwrapped

N_TIMES = 1
N_EP = 3000
N_STEP = 200

N_WARMUP_EP = 20
N_REPLACE_TARGET = 200

N_MEM = 30000
N_BATCH = 32

ACT_VAR = 1
ACT_VAR_DEST = 0.01
ACT_VAR_STEP = 50 * N_STEP
ACT_VAR_DECAY = pow(ACT_VAR_DEST / ACT_VAR, 1 / ACT_VAR_STEP)
epsilon = 0.1

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high[0]
print(a_bound)

## log files

now = datetime.datetime.now()
basename = os.path.basename(__file__)
LOG_PATH = 'logs/%s' % (basename)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
LOG_PATH = 'logs/%s/%s' % (basename, now.strftime('%Y_%m_%d_%H_%M_%S'))
os.mkdir(LOG_PATH)
LOG_FILE = os.path.join(LOG_PATH, 'train.log')
log_fp = open(LOG_FILE, 'w')
print(LOG_FILE)

pyfile = open(__file__)
bakfile = open(os.path.join(LOG_PATH, basename), 'w')
bakfile.writelines(pyfile.readlines())
pyfile.close()
bakfile.close()

save_rewards = np.zeros((N_TIMES, N_EP))

ddpg = DDPG(s_dim, a_dim)
mem = ReplayMemory(s_dim, a_dim, N_MEM)

tqdm = lambda x: x

for t in range(N_TIMES):
    print("Time: %d" % t)
    ddpg.reset()
    mem.reset()
    total_step = 0
    train_step = 0
    for ep in tqdm(range(N_EP)):
        ep_reward = 0
        step = 0
        state = env.reset()
        for step in range(N_STEP):
            env.render()
            if np.random.uniform() < epsilon:
                action = np.random.uniform()
            else:
                action = ddpg.choose_action(state)[0] + np.random.normal(0, ACT_VAR)
                action = np.clip(action, 0, 1)
            act = action * a_bound * 2 - a_bound
            state_next, reward, done, _ = env.step([act])
            
            # print(state, state_next, action, reward, act)
            mem.store_transition(state, state_next, action, reward)
            
            if ep > N_WARMUP_EP:
                ddpg.learn(mem, N_BATCH)
                train_step += 1
                ACT_VAR *= ACT_VAR_DECAY
                
            state = state_next
            ep_reward += reward
            total_step += 1
            
        report = "E: {ep} T: {ts} V: {var} R: {reward}\n".format(
            ep=ep,
            step=step,
            reward=ep_reward,
            ts=total_step,
            var=ACT_VAR)
        log_fp.write(report)
        log_fp.flush()
        print(report)
        save_rewards[t, ep] = reward

step_fn = os.path.join(LOG_PATH, 'rewards.mat')
sio.savemat(step_fn, {'rewards': save_rewards})

plt.plot(save_rewards.mean(0))
plt.xlabel('Time slot')
plt.ylabel('Rewards')
plt.show()

print(LOG_PATH)
