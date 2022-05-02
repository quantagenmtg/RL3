# %%
import numpy as np
import torch
from torch import FloatTensor,Tensor,LongTensor
from Policy import REINFORCE, ActorCritic
import gym
import random
import matplotlib.pyplot as plt
import time
import argparse


# %% [markdown]
# # REINFORCE
# We define a function to run REINFORCE algorithm on.

# %%
def Cartpole(total_episodes, learning_rate, future_reward_discount_factor, hidden_shape):
    """
    Tries to solve Cartpole-v1 usinf the REINFORCE algorithm. Right now it only applies a Monte-Carlo REINFORCE

    Args:
        total_episodes: How many times the environment resets
        learning_rate: For optimizer
        future_reward_discount_factor: future rewards are dicounted
        hidden_shape: List of integers. [16,16] would give two hidden layers (linear with PReLU activation) with both 16 nodes in the policy model
    
    Returns:
        scores: Score per episode in a list
    """
    
    
    scores = []
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env.observation_space.shape[0], env.action_space.n, learning_rate, future_reward_discount_factor, hidden_shape)

    for i in range(total_episodes):
        #reset the environment
        state = env.reset()
        rewards = []
        log_probs = []

        #Cartpole-v1 has a maximum episode length of 500
        for t in range(500):
            #env.render()
            #Action selection is done by the policy
            action, log_prob = agent.model.pick(state)

            #Get example
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)

            #The score is how long the cart stayed upright, this can be a maximum of 500
            if done or t==499:
                print(f"Episode {i}: Score {t+1}/500")
                break
        
        rewards = torch.tensor(rewards)
        log_probs = torch.cat(log_probs)
        agent.update(rewards,log_probs)

        scores.append(t+1)

        #Cartpole v1 is considered solved when
        # the average over the last 100 consecutive episodes is at least 475
        if np.mean(scores[-100:]) >= 475: 
            mask = np.full(total_episodes - (i + 1),500)
            scores = np.concatenate((scores,mask))
            print("Solved!")
            break

    if np.mean(scores[-100:]) < 475: 
        print("Not Solved...")
    return scores

    

# %% [markdown]
# # Actor-Critic

# %%
def AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic):
    """
    Tries to solve Cartpole-v1 usinf the REINFORCE algorithm. Right now it only applies a Monte-Carlo REINFORCE

    Args:
        total_episodes: How many times the environment resets
        learning_rate: For optimizer
        future_reward_discount_factor: future rewards are dicounted
        hidden_shape: List of integers. [16,16] would give two hidden layers (linear with PReLU activation) with both 16 nodes in the policy model
    
    Returns:
        scores: Score per episode in a list
    """
    
    
    scores = []
    env = gym.make("CartPole-v1")
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n, estimation_depth, gradient_method, learning_rate, hidden_shape_actor, hidden_shape_critic)

    for i in range(total_episodes):
        #reset the environment
        state = env.reset()
        rewards = []
        log_probs = []
        states = []

        #Cartpole-v1 has a maximum episode length of 500
        for t in range(500):
            #env.render()
            states.append(state)

            #Action selection is done by the policy
            action, log_prob = agent.actor.pick(state)

            #Get example
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            #The score is how long the cart stayed upright, this can be a maximum of 500
            if done or t==499:
                print(f"Episode {i}: Score {t+1}/500")
                break

        states = torch.Tensor(np.array(states))
        rewards = np.array(rewards)
        log_probs = torch.cat(log_probs)
        agent.update(rewards,log_probs, states)

        scores.append(t+1)
        #Cartpole v1 is considered solved when
        # the average over the last 100 consecutive episodes is at least 475
        if np.mean(scores[-100:]) >= 475: 
            mask = np.full(total_episodes - (i + 1),500)
            scores = np.concatenate((scores,mask))
            print("Solved!")
            break
    if np.mean(scores[-100:]) < 475: 
        print("Not Solved...")
    return scores

    

# %%
def plot_results(total_episodes, score):
    #Plot score per episode
    plt.plot(np.arange(1,total_episodes+1), score)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()

# %%
def run_experiments(method, total_episodes = 1000, learning_rate = 1e-2, future_discount = 1, estimation_depth = 500, gradient_method = 'both', hidden_shape = 32, hidden_shape_actor = 16, hidden_shape_critic = 16, hidden_layers = 1, hidden_layers_actor = 1, hidden_layers_critic = 1):
    if method == "AC":
        score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, [hidden_shape_actor for _ in range(hidden_layers_actor)], [hidden_shape_critic for _ in range(hidden_layers_critic)])
    if method == "REINFORCE":
        score = Cartpole(total_episodes, learning_rate, future_discount, [hidden_shape for _ in range(hidden_layers)])
    plot_results(total_episodes, score)

# %%
#run_experiments("AC")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='Method', dest = 'method', required = True)
    parser.add_argument('-eps', help='Total episodes', dest = 'episodes', required=False)
    parser.add_argument('-lr', help='Learning rate', dest = 'learning_rate', required=False)
    parser.add_argument('-d', help='Future reward discount factor (REINFORCE)', dest = 'discount', required=False)
    parser.add_argument('-est', help='Estimation depth (AC)', dest = 'estimation', required=False)
    parser.add_argument('-grad', help='Gradient method (AC)', dest = 'gradient', required=False)
    parser.add_argument('-hs', help='Hidden shape (REINFORCE)', dest = 'hidden', required=False)
    parser.add_argument('-ha', help='Hidden shape actor (AC)', dest = 'hidden_actor', required=False)
    parser.add_argument('-hc', help='Hidden shape critic (AC))', dest = 'hidden_critic', required=False)
    parser.add_argument('-l', help='Hidden layers (REINFORCE)', dest = 'layers', required=False)
    parser.add_argument('-la', help='Hidden layers actor (AC)', dest = 'layers_actor', required=False)
    parser.add_argument('-lc', help='Hidden layers critic (AC)', dest = 'layers_critic', required=False)
    args = parser.parse_args()
    kwargs = dict(total_episodes=args.episodes,learning_rate=args.learning_rate,future_discount=args.discount,estimation_depth=args.estimation,
    gradient_method=args.gradient,hidden_shape=args.hidden,hidden_shape_actor=args.hidden_actor,hidden_shape_critic=args.hidden_critic,
    hidden_layers=args.layers,hidden_layers_actor=args.layers_actor,hidden_layers_critic=args.layers_critic)
    
    run_experiments(args.method, **{k: v for k, v in kwargs.items() if v is not None})

if __name__ == "__main__":
    main()
