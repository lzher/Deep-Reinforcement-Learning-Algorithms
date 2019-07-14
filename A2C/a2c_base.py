import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
import math

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class ReplayMemory:
    def __init__(self, n_states, dim_actions, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.memory = torch.zeros([capacity, n_states*2+dim_actions+1]).to(device)

    def store_transition(self, state, state_next, action, reward):
        idx = self.ptr % self.capacity
        self.memory[idx, :] = torch.from_numpy(np.hstack([state, state_next, action, reward]))
        self.ptr += 1

    def sample_transitions(self, sample_size):
        idx = np.random.choice(np.min([self.capacity, self.ptr]), size=sample_size)
        return self.memory[idx]

    def reset(self):
        self.ptr = 0
        
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, units=[200, 100]):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.fc_1 = nn.Linear(s_dim, units[0])
        self.fc_2 = nn.Linear(units[0], units[1])
        self.fc_mu = nn.Linear(units[1], a_dim)
        self.fc_sig = nn.Linear(units[1], a_dim)
        
        init_layers = [self.fc_1, self.fc_2, self.fc_mu, self.fc_sig]
        for layer in init_layers:
            nn.init.normal_(layer.weight, mean = 0, std = 0.1)
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        mu = F.tanh(mu) * 2
        sigma = self.fc_sig(x)
        sigma = F.softplus(sigma) + 0.001
        return mu, sigma
        
class Critic(nn.Module):
    def __init__(self, s_dim, units=[200, 100]):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.fc_1 = nn.Linear(s_dim, units[0])
        self.fc_2 = nn.Linear(units[0], units[1])
        self.fc_value = nn.Linear(units[1], 1)
        
        init_layers = [self.fc_1, self.fc_2, self.fc_value]
        for layer in init_layers:
            nn.init.normal_(layer.weight, mean = 0, std = 0.1)
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_value(x)
        return x
        
        
class A2C:
    def __init__(self, s_dim, a_dim, a_units=[40, 20], c_units=[40, 20], gamma=0.9, \
                  a_lr=2e-4, c_lr=2e-4):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma
        
        self.actor = Actor(s_dim, a_dim, a_units).to(device)
        self.critic = Critic(s_dim, c_units).to(device)
        
        self.distribution = torch.distributions.Normal
        
        # self.a_op = optim.Adam(self.actor.parameters(), lr=a_lr)
        # self.c_op = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.train_op = optim.Adam(itertools.chain(self.actor.parameters(),
                                                   self.critic.parameters()),
                                   lr=a_lr)
        self.target_values = None
        
    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        mu, sigma = self.actor(state)
        m = self.distribution(mu, sigma)
        action = m.sample().detach().cpu().numpy()
        return action
        
    def learn_episode(self, ep_mem, done):
        steps = ep_mem.ptr
        batch_memory = ep_mem.memory[:steps]
        batch_states = batch_memory[:, 0:self.s_dim]
        batch_states_next = batch_memory[:, self.s_dim:self.s_dim*2]
        batch_actions = batch_memory[:, self.s_dim*2:self.s_dim*2+self.a_dim]
        batch_rewards = batch_memory[:, self.s_dim*2+self.a_dim:self.s_dim*2+self.a_dim+1]
        
        # print(batch_memory, batch_states, batch_states_next, batch_actions, batch_rewards)
        
        if self.target_values is None:
            self.target_values = torch.zeros((ep_mem.memory.shape[0], 1)).to(device)
        
        if done:
            value = 0
        else:
            value = self.critic(batch_states_next[-1])
        target_values = self.target_values[:steps]
        for i in range(len(batch_rewards))[::-1]:
            value = batch_rewards[i] + self.gamma * value
            target_values[i] = value
        target_values = target_values.detach()
            
        batch_values = self.critic(batch_states)
        batch_td = target_values - batch_values
        c_loss = batch_td.pow(2)
        # c_loss = c_loss.mean()
        
        mu, sigma = self.actor(batch_states)
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(batch_actions)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        a_loss = -log_prob * batch_td.detach() - 0.005 * entropy
        # a_loss = a_loss.mean()
        
        # self.c_op.zero_grad()
        # c_loss.backward()
        # self.c_op.step()
        # self.a_op.zero_grad()
        # a_loss.backward()
        # self.a_op.step()
        
        loss = (a_loss + c_loss).mean()
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        
        return c_loss, a_loss
        
        
    def learn_step(self, state, state_next, action, reward):
        state = torch.from_numpy(state).float().to(device)
        state_next = torch.from_numpy(state_next).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.FloatTensor([reward]).to(device)
        
        value = self.critic(state)
        value_next = self.critic(state_next)
        value_target = reward + self.gamma * value_next
        td = value_target - value
        c_loss = td.pow(2)
        
        self.c_op.zero_grad()
        c_loss.backward()
        self.c_op.step()
        
        mu, sigma = self.actor(state)
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(action)
        a_loss = -log_prob * td.detach()
        
        self.a_op.zero_grad()
        a_loss.backward()
        self.a_op.step()
        
    def learn(self, mem, batch_size):
        batch_memory = mem.sample_transitions(batch_size)
        batch_states = batch_memory[:, 0:self.s_dim]
        batch_states_next = batch_memory[:, self.s_dim:self.s_dim*2]
        batch_actions = batch_memory[:, self.s_dim*2:self.s_dim*2+self.a_dim]
        batch_rewards = batch_memory[:, self.s_dim*2+self.a_dim:self.s_dim*2+self.a_dim+1]
        
        mu, sigma = self.actor(batch_states)
        value = self.critic(batch_states)
        value_next = self.critic(batch_states_next)
        value_target = batch_rewards + self.gamma * value_next
        
        td = value_target - value
        td = td.mean()
        c_loss = td.pow(2)
        
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(batch_actions).mean()
        a_loss = -log_prob * td.detach()
        
        self.c_op.zero_grad()
        c_loss.backward()
        self.c_op.step()
        self.a_op.zero_grad()
        a_loss.backward()
        self.a_op.step()
        
        
        
        
        
        
        
        
        
        
