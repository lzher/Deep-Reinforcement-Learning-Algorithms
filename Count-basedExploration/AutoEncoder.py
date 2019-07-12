import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Use CUDA')
    device = torch.device('cuda:0')
else:
    print('Use CPU')
    device = torch.device('cpu')
    
class AEModel(nn.Module):
    def __init__(self, fc_units, encode_layer='mid', activators='leaky'):
        super(AEModel, self).__init__()
        self.fc_units = fc_units
        self.n_layers = len(fc_units) - 1
        self.fc_layers = [nn.Linear(fc_units[i], fc_units[i+1]).to(device) for i in range(self.n_layers)]
        if activators == 'leaky':
            self.activators = [nn.LeakyReLU().to(device) for i in range(self.n_layers - 1)]
        else:
            self.activators = activators
        if encode_layer == 'mid':
            self.encode_layer = self.n_layers // 2
        else:
            self.encode_layer = encode_layer
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.fc_layers[i](x)
            if i < self.n_layers-1:
                x = self.activators[i](x)
        return x
    
    def encode(self, x):
        for i in range(self.encode_layer):
            x = self.fc_layers[i](x)
            if i < self.encode_layer-1:
                x = self.activators[i](x)
        return x
        
    def decode(self, x):
        for i in range(self.encode_layer, self.n_layers):
            x = self.fc_layers[i](x)
            if i < self.n_layers-1:
                x = self.activators[i](x)
        return x
        
    def parameters(self):
        params = []
        for i in range(self.n_layers):
            params.extend(list(self.fc_layers[i].parameters()))
        return params

class AutoEncoder:
    def __init__(self, fc_units, LR=1e-3):
        self.model = AEModel(fc_units)
        self.loss_fn = nn.MSELoss().to(device)
        self.train_op = optim.Adam(self.model.parameters(), lr=LR)
        
    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, x):
        return self.model.decode(x)
    
    def learn(self, batch_memory):
        x = self.model.forward(batch_memory)
        
        loss = self.loss_fn(x, batch_memory)
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        
        return loss.detach().cpu().numpy()

if __name__ == '__main__':
    batch_size = 32
    training_steps = 1000000
    
    units = [200, 100, 20, 2, 20, 100, 200]
    ae = AutoEncoder(units)
    
    t = np.linspace(0, 1, 200)
    for step in range(training_steps):
        x = np.random.uniform(0, np.pi, size=(batch_size,1))
        w = np.random.uniform(0, 10, size=(batch_size,1))
        y = torch.from_numpy(np.sin(2 * np.pi * w * t + x)).float().to(device)
        loss = ae.learn(y)
        print("S: {s}/{st} L: {l}".format(s=step, st=training_steps, l=loss))
        
    torch.save(ae, 'ae.bin')
    
    x = 1
    w = 5
    y = torch.from_numpy(np.sin(2 * np.pi * w * t + x)).float().to(device)
    print(ae.encode(y))
    
    z = ae.decode(torch.FloatTensor([5, 1]).to(device))
    print(z)
    
    plt.plot(z.detach().cpu().numpy())
    plt.plot(np.sin(2 * np.pi * 5 * t + 1))
    plt.plot(np.sin(2 * np.pi * 1 * t + 5))
    plt.show()
        