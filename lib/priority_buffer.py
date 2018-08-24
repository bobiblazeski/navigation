import numpy as np
import random
import torch

from .sum_tree import SumTree

class PriorityBuffer:
    '''
    Almost copy from
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    '''
    def __init__(self, max_size, batch_size, device):
        self.tree = SumTree(max_size)
        self._max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.e = 0.01
        self.a = 0.6
 
    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, old_state, action, reward, new_state, is_terminal):
        # 0.5 is the maximum error
        p = self._getPriority(0.5)
        self.tree.add(p, data=(old_state, action, reward, new_state, is_terminal)) 

    def sample(self, indexes=None):
        data_batch = []
        idx_batch = []
        segment = self.tree.total_and_count()[0] / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
        
        states = torch.from_numpy(np.vstack([e[0] for e in data_batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in data_batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in data_batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in data_batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in data_batch if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones), idx_batch

    def update(self, idx_list, error_list):
        for idx, error in zip(idx_list, error_list):
            p = self._getPriority(error)
            self.tree.update(idx, p)
            
    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.count