
import numpy as np
import random

import torch
from torch import distributions
import torch.nn as nn
from torch.nn.modules.activation import Softmax
from torch.distributions import Categorical
import torch.optim as optim
from  torch.autograd import Variable

import time
from collections import namedtuple

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_shape) -> None:
        """
            Policy network. Gives probabilities of picking actions.
        """
        super(Policy, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_shape[0]),
            nn.PReLU()
        )

        self.layers = []
        for i,n in enumerate(hidden_shape[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n, hidden_shape[i+1]),
                    nn.PReLU()
                )
            )

        self.output = nn.Sequential(
            nn.Linear(hidden_shape[-1], output_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.input(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x
    
    def pick(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        
        return action.item(), distribution.log_prob(action)


class REINFORCE():
    def __init__(self, state_shape, n_possible_actions, learning_rate = 0.01, future_reward_discount_factor = 0.95, hidden_shape = [16,16]):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.learning_rate = learning_rate
        self.gamma = future_reward_discount_factor
        self.model = Policy(state_shape, n_possible_actions, hidden_shape)
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def update(self, rewards, log_probs):
        #calculate discount for each rewards
        discounts = torch.ones(rewards.size(dim=0))*self.gamma
        discounts = torch.cumprod(discounts, dim=0)/self.gamma

        #calculate discounted sum of rewards
        discrew = discounts*rewards
        discrew = discrew + torch.sum(discrew,dim = 0, keepdims=True) - torch.cumsum(discrew, dim=0)

        #To turn gradient descent into ascent we multipply by -1
        loss= (-log_probs * discrew).sum()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()