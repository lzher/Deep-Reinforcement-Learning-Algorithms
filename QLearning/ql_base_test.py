from ql_base import QLearning
import numpy as np
import matplotlib.pyplot as plt

N_STATE = 5
ql = QLearning(N_STATE, N_STATE)

# env = np.random.normal(size=(N_STATE, N_STATE))
env = np.array([[-0.48941831, 1.92637755, 2.06612145,-0.45121798, 0.74563957],
 [-1.37278796,-0.79193798,-0.12773108, 3.11084087, 1.66773807],
 [-3.19554563, 0.68490444, 1.11000035, 1.23610048,-0.28320787],
 [-0.00447734, 0.46827985,-0.52582707,-2.10689542, 1.42040937],
 [ 1.76729681, 0.3559686, -0.22965815, 0.55411405, 0.3694959,],],)
print(env)

N_STEP = 100000
state = np.random.randint(N_STATE)

Save_Reward = np.zeros((N_STEP))

for step in range(N_STEP):
    action = ql.choose_action(state)
    state_next = action
    reward = env[state, state_next]
    ql.learn(state, state_next, action, reward)
    print("S: {s} T: {t} A: {a} R: {r}".format(s=step, t=state, a=action, r=reward))
    
    state = state_next
    Save_Reward[step] = reward
    
print(env)
print(ql.qtable)
    
# plt.plot(Save_Reward)
# plt.show()
