from dqn_base import DQN, ReplayMemory
import gym
import datetime
import os

env = gym.make('CartPole-v1').unwrapped

N_EP = 300

N_WARMUP = 200
N_REPLACE_TARGET = 200

N_MEM = 30000
N_BATCH = 32

s_dim = 4
a_dim = 1
a_val = 2

now = datetime.datetime.now()
LOG_PATH = 'logs/R_' + os.path.basename(__file__) + now.strftime('_%Y_%m_%d_%H_%M_%S')
os.mkdir(LOG_PATH)
LOG_FILE = LOG_PATH + '/train.log'
log_fp = open(LOG_FILE, 'w')
print(LOG_FILE)

dqn = DQN(s_dim, a_val, a_dim)
mem = ReplayMemory(s_dim, a_dim, N_MEM)

total_step = 0
train_step = 0
for ep in range(N_EP):
    ep_reward = 0
    step = 0
    state = env.reset()
    while True:
        # env.render()
        action = dqn.choose_action(state)
        state_next, reward, done, _ = env.step(action)
        
        xpos, xvel, theta, thetavel = state_next
        r1 = (env.x_threshold - abs(xpos)) / env.x_threshold - 0.5
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        
        mem.store_transition(state, state_next, action, reward)
        
        if total_step > N_WARMUP:
            dqn.learn(mem, N_BATCH, replacement=(train_step % N_REPLACE_TARGET == 0))
            train_step += 1
            
        state = state_next
        ep_reward += reward
        step += 1
        total_step += 1
        
        if done:
            report = "E: {ep} S: {step} T: {ts} R: {reward}\n".format(
                ep=ep,
                step=step,
                reward=ep_reward,
                ts=total_step)
            log_fp.write(report)
            log_fp.flush()
            print(report)
            break






