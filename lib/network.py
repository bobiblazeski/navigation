import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        return self.fc2(x)
   
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1_adv = nn.Linear(state_size, fc1_units)
        self.fc1_val = nn.Linear(state_size, fc1_units)
        self.fc2_adv = nn.Linear(fc1_units, action_size)
        self.fc2_val = nn.Linear(fc1_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        batch_size = state.size(0)
        adv = F.relu(self.fc1_adv(state))
        val = F.relu(self.fc1_val(state))
        adv = self.fc2_adv(adv)
        x_size = batch_size
        val = self.fc2_val(val).expand(x_size, self.action_size)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x_size, self.action_size)
        return x