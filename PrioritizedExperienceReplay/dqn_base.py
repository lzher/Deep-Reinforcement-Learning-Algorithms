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

class DQN_model(nn.Module):
    def __init__(self, n_states, n_actions, n_units = [200, 100]):
        super(DQN_model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc_1 = nn.Linear(n_states, n_units[0])
        self.fc_2 = nn.Linear(n_units[0], n_units[1])
        self.fc_value = nn.Linear(n_units[1], n_actions)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)
        x = F.leaky_relu(x)
        x = self.fc_value(x)
        return x
        
    def reset(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_value.reset_parameters()
        
        
class DQN:
    def __init__(self, n_states, n_actions, action_dim, n_units = [200, 100],
                 gamma = 0.9, epsilon = 0.1, LR=1e-4):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.eval = DQN_model(n_states, n_actions, n_units=n_units).to(device)
        self.target = DQN_model(n_states, n_actions, n_units=n_units).to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.train_op = optim.Adam(self.eval.parameters(), lr=LR)
        
        self.replace_target()
        
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = torch.from_numpy(np.array([state])).float().to(device)
            Q_state = self.eval(state).detach().cpu().numpy()
            action = np.argmax(Q_state)
        return action
        
    def replace_target(self):
        self.target.load_state_dict(self.eval.state_dict())
        
    def learn(self, mem, batch_size, replacement=False):
        if replacement:
            self.replace_target()
        batch_memory = mem.sample_transitions(batch_size)
        batch_states = batch_memory[:, 0:self.n_states]
        batch_states_next = batch_memory[:, self.n_states:self.n_states*2]
        batch_actions = batch_memory[:, self.n_states*2:self.n_states*2+self.action_dim].long()
        batch_rewards = batch_memory[:, self.n_states*2+self.action_dim:self.n_states*2+self.action_dim+1]
        
        q_eval = self.eval(batch_states)
        q_eval = q_eval.gather(1, batch_actions)
        
        q_target = self.target(batch_states_next).detach()
        q_target = q_target.max(1)[0]
        q_target = q_target.reshape(-1, 1)
        q_target = batch_rewards + self.gamma * q_target
        
        loss = self.mse_loss(q_eval, q_target)
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        
    def reset(self):
        self.eval.reset()
        self.target.reset()
        self.replace_target()
        
        
        
        
        
    