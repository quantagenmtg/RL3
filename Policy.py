
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


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_shape) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(Critic, self).__init__()

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

        self.output = torch.nn.Linear(hidden_shape[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)            
        """
        
        x = self.input(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x


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
        discrew = discrew/discounts

        #To turn gradient descent into ascent we multiply by -1
        loss= (-log_probs * discrew).sum()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


class ActorCritic():
    def __init__(self, state_shape, n_possible_actions, estimation_depth, gradient_method = 'nstep', learning_rate = 0.01, hidden_shape_actor = [16,16], hidden_shape_critic = [16,16]):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.n = int(estimation_depth)
        self.method = gradient_method
        self.learning_rate = learning_rate
        self.actor = Policy(state_shape, n_possible_actions, hidden_shape_actor)
        self.critic = Critic(state_shape, hidden_shape_critic)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def update(self, rewards, log_probs, states):
        
        #n-step reward calculation
        cumsum1 = np.cumsum(rewards[::-1])[::-1]
        cumsum2 = np.roll(cumsum1,-self.n)
        cumsum2[-self.n:] = 0
        n_step_rewards = cumsum1 - cumsum2

        #n-step target and output calculation
        id = torch.ones(rewards.shape[0])
        id[-self.n:] = 0
        #why does it work better without.flatten()????????
        n_step_target = torch.Tensor(n_step_rewards) + id*self.critic(states.roll(-self.n)).flatten()
        V_expected = self.critic(states).flatten()

        #loss for the critic NN
        self.optim_critic.zero_grad()
        loss_critic = ((V_expected - n_step_target)**2).sum()
        
        # backpropagation of loss to critic NN   
        loss_critic.backward()
        self.optim_critic.step()

        #loss for the actor NN
        self.optim_actor.zero_grad()
        if self.method == 'nstep':
            loss_actor = (-log_probs * n_step_target.detach()).sum()
        if self.method == 'baseline':
            baseline = torch.Tensor(np.copy(cumsum1)) - V_expected
            loss_actor = (-log_probs * baseline.detach()).sum()
        if self.method == 'both':
            both = n_step_target - V_expected
            loss_actor = (-log_probs * both.detach()).sum()
        
        # backpropagation of loss to actor NN   
        loss_actor.backward()
        self.optim_actor.step()