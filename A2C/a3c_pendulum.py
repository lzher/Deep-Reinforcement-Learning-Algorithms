import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import math
import gym
import matplotlib.pyplot as plt
import ctypes

N_DENSE     =   100
MAX_EP      =   5000
MAX_EP_STEP =   50
UPDATE_STEP =   5
ADAM_LR     =   0.0002

GAMMA       =   0.9

env = gym.make('Pendulum-v0').unwrapped
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.a_fc = nn.Linear(s_dim, N_DENSE)
        self.a_mu = nn.Linear(N_DENSE, a_dim)
        self.a_sigma = nn.Linear(N_DENSE, a_dim)

        self.c_fc = nn.Linear(s_dim, N_DENSE)
        self.c_value = nn.Linear(N_DENSE, 1)

        init_layers = [self.a_fc, self.a_mu, self.a_sigma, self.c_fc, self.c_value]
        for layer in init_layers:
            nn.init.normal_(layer.weight, mean = 0, std = 0.1)
            nn.init.constant_(layer.bias, 0.1)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu(self.a_fc(x))
        mu = torch.tanh(self.a_mu(a1)) * 2
        sigma = F.softplus(self.a_sigma(a1)) + 0.001
        c1 = F.relu(self.c_fc(x))
        values = self.c_value(c1)
        return mu, sigma, values
        
    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        return m.sample().numpy()

    def loss_func(self, s, a, value_target):
        self.train()

        mu, sigma, values = self.forward(s)
        td = value_target - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        exp_v = log_prob * td.detach()
        a_loss = -exp_v

        exploration = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exploration = -0.005 * exploration
        a_loss += exploration
        
        total_loss = (a_loss + c_loss).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, g_net, opt, g_ep, g_rewards, wi):
        super(Worker, self).__init__()
        self.g_net = g_net
        self.opt = opt
        self.g_ep = g_ep
        self.wi = wi
        self.l_net = Net(N_S, N_A)
        self.env = gym.make('Pendulum-v0').unwrapped
        self.g_rewards = g_rewards

    def run(self):
        save_rewards = np.frombuffer(self.g_rewards.get_obj())
        total_step = 0
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            bs, ba, br = [], [], []
            ep_r = 0.0
            for step in range(MAX_EP_STEP):
                # if self.wi == 0:
                    # self.env.render()
                a = self.l_net.choose_action(vwrap(s)).clip(-2, 2)
                s_, r, d, _ = self.env.step(a)

                d = (step == MAX_EP_STEP - 1)
                r = (r + 8.1) / 8.1

                bs.append(s)
                ba.append(a)
                # br.append((r + 8.1) / 8.1) # key factor: normalization
                br.append(r)
                
                ep_r += r
                total_step += 1
                if (d) or (total_step % UPDATE_STEP == 0):
                    update_net(self.l_net, self.g_net, self.opt, d, s_, bs, ba, br)
                    bs, ba, br = [], [], []
                    if d:
                        with self.g_ep.get_lock():
                            self.g_ep.value += 1
                        print("W:%d E:%d R:%f" % (self.wi, self.g_ep.value, ep_r))
                        save_rewards[self.g_ep.value] = ep_r
                s = s_

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0):
        super(SharedAdam, self).__init__(params,
                                         lr = lr,
                                         betas = betas,
                                         eps = eps,
                                         weight_decay = weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def vwrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def update_net(l_net, g_net, opt, done, s_, bs, ba, br):
    if done:
        value_ = 0.0
    else:
        value_ = l_net.forward(vwrap(s_))[-1].data.numpy()[0]

    value_target = []
    for r in br[::-1]:
        value_ = r + GAMMA * value_
        value_target.append(value_)
    value_target.reverse()
    # value_target = np.array(value_target)
    # value_target = (value_target - value_target.min()) / (value_target.max() - value_target.min())

    loss = l_net.loss_func(
        vwrap(np.vstack(bs)),
        vwrap(np.vstack(ba)),
        vwrap(np.vstack(value_target)))

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(l_net.parameters(), g_net.parameters()):
        gp._grad = lp.grad
    opt.step()

    l_net.load_state_dict(g_net.state_dict())

if __name__ == "__main__":
    g_net = Net(N_S, N_A)
    g_net.share_memory()
    opt = SharedAdam(g_net.parameters(), lr = ADAM_LR)
    g_ep = mp.Value('i', 0)
    g_rewards = mp.Array(ctypes.c_double, MAX_EP * 2)

    workers = [Worker(g_net, opt, g_ep, g_rewards, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]
    save_rewards = np.frombuffer(g_rewards.get_obj())
    plt.plot(save_rewards[:g_ep.value])
    plt.show()
