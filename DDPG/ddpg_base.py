import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class ReplayMemory:
    def __init__(self, n_states, n_actions, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.memory = torch.zeros([capacity, n_states*2+n_actions+1]).to(device)

    def store_transition(self, state, state_next, action, reward):
        idx = self.ptr % self.capacity
        self.memory[idx, :] = torch.from_numpy(np.hstack([state, state_next, action, reward]))
        self.ptr += 1

    def sample_transitions(self, sample_size):
        if self.ptr == 0:
            return torch.zeros(0)
        idx = np.random.choice(np.min([self.capacity, self.ptr]), size=sample_size)
        return self.memory[idx]

    def reset(self):
        self.ptr = 0

class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions, n_units = [20, 10], init_w=1e-5):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc_act = nn.Linear(n_units[1], n_actions)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc_act.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x, train=True):
        # x = x.view(x.size(0), -1, x.size(-1))
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.leaky_relu(x)
        # x = x.view(x.size(0), -1)
        x = self.fc_act(x)
        x = F.sigmoid(x)
        return x
        
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc_act.reset_parameters()

class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions, n_units = [5, 20, 10], init_w=1e-5):
        super(CriticNet, self).__init__()
        self.fc_s = nn.Linear(n_states, n_units[0])
        # self.fc_a = nn.Linear(n_actions, n_units[0])
        self.fc1 = nn.Linear(n_units[0] + n_actions, n_units[1])
        self.fc2 = nn.Linear(n_units[1], n_units[2])
        self.fc_value = nn.Linear(n_units[2], 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        
        self.fc_s.weight.data.uniform_(-init_w, init_w)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc_value.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, s, a, train=True):
        s = self.fc_s(s)
        s = F.leaky_relu(s)
        # a = self.fc_a(a)
        x = torch.cat([s, a], 1)
        # x = x.view(x.size(0), -1, x.size(-1))
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.leaky_relu(x)
        # x = x.view(x.size(0), -1)
        x = self.fc_value(x)
        return x
        
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc_s.reset_parameters()
        self.fc_value.reset_parameters()
        
        
class DDPG:
    def __init__(self, n_states, n_actions, gamma=0.99, \
                 tau=1e-3, actor_lr=4e-4, critic_lr=8e-4, epsilon=0.1,
                 critic_units=[5,20,10], actor_units=[20,10]):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon / n_actions
        
        self.actor_eval = ActorNet(n_states, n_actions).to(device)
        self.actor_target = ActorNet(n_states, n_actions).to(device)
        self.critic_eval = CriticNet(n_states, n_actions).to(device)
        self.critic_target = CriticNet(n_states, n_actions).to(device)
        
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        
        self.actor_optim = optim.Adam(self.actor_eval.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic_eval.parameters(), lr=critic_lr)
        self.critic_loss_fn = nn.MSELoss()
        
    def choose_action(self, state):
        state = torch.from_numpy(np.array([state])).float().to(device)
        action = self.actor_eval(state, train=False).detach().cpu().numpy()
        action = action.squeeze(0)
        return action
        
    def replace_target(self):
        # Hard replacement employed first
        # self.actor_target.load_state_dict(self.actor_eval.state_dict())
        # self.critic_target.load_state_dict(self.critic_eval.state_dict())
        for actor_target_param, actor_eval_param in \
            zip(self.actor_target.parameters(), self.actor_eval.parameters()):
            actor_target_param.data.copy_((1-self.tau) * actor_target_param.data + \
                                           self.tau * actor_eval_param.data)
        for critic_target_param, critic_eval_param in \
            zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            critic_target_param.data.copy_((1-self.tau) * critic_target_param.data + \
                                            self.tau * critic_eval_param.data)
    
    def learn(self, mem, batch_size):
        batch_memory = mem.sample_transitions(batch_size)
        batch_states = batch_memory[:, 0:self.n_states]
        batch_states_next = batch_memory[:, self.n_states:self.n_states*2]
        batch_actions = batch_memory[:, self.n_states*2:self.n_states*2+self.n_actions]
        batch_rewards = batch_memory[:, self.n_states*2+self.n_actions:self.n_states*2+self.n_actions+1]
        # print(batch_memory[1])
        # print(batch_states[1], batch_states_next[1], batch_actions[1], batch_rewards[1])
        
        Q_eval = self.critic_eval(batch_states, batch_actions)
        Q_next = self.critic_target(batch_states_next, self.actor_target(batch_states_next))
        Q_target = batch_rewards + self.gamma * Q_next
        Q_target = Q_target.detach()
        
        critic_loss = self.critic_loss_fn(Q_eval, Q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        actor_loss = -torch.mean(self.critic_eval(batch_states, self.actor_eval(batch_states)))
        # print(actor_loss)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.replace_target()
        
    def reset(self):
        self.actor_eval.reset()
        self.actor_target.reset()
        self.critic_eval.reset()
        self.critic_target.reset()
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        


