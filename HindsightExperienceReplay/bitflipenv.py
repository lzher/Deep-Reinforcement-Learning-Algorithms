import numpy as np

class BitFlipEnv:
    def __init__(self, nbits=10):
        self.nbits = nbits
        self.reset(nbits)
        
    def reset(self, nbits=None):
        if nbits == None:
            nbits = self.nbits
        self.state = np.random.randint(2, size=nbits)
        self.target = np.random.randint(2, size=nbits)
        return [self.state, self.target]
        
    def step(self, flip_bit):
        self.state[flip_bit] = 1 - self.state[flip_bit]
        reward = int(np.array_equal(self.state, self.target))
        return [self.state, self.target], reward, (reward == 1), ''