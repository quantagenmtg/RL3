# %%
import numpy as np
import torch
import gym
import random
import matplotlib.pyplot as plt
import time
import argparse
import pickle

from torch import FloatTensor,Tensor,LongTensor
from Policy import REINFORCE, ActorCritic
from os.path import exists
from pathlib import Path
from tqdm import tqdm
from operator import itemgetter
# %% [markdown]
# # REINFORCE
# We define a function to run REINFORCE algorithm on.

# %%
def Cartpole(total_episodes, learning_rate, future_reward_discount_factor, hidden_shape, silent=True):
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
    learned_at = total_episodes
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
                if not silent:
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
            learned_at = i
            if not silent:
                print("Solved!")
            break

    if np.mean(scores[-100:]) < 475: 
        if not silent:
            print("Not Solved...")
    return scores, learned_at

    

# %% [markdown]
# # Actor-Critic

# %%
def AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic, silent = True):
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
    learned_at = total_episodes
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
                if not silent:
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
            learned_at = i
            if not silent:
                print("Solved!")
            break
    if np.mean(scores[-100:]) < 475:
        if not silent: 
            print("Not Solved...")
    return scores, learned_at

    

# %%
def plot_results(total_episodes, score):
    #Plot score per episode
    plt.plot(np.arange(1,total_episodes+1), score)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()

# %%
def retrieve_dict(key, frozen_dict):
    for key, value in frozen_dict.items():
        print(key)
        exit(0)
        
def keywithmaxval(d, idx):
    """ a) create a list of the dict's keys and values; 
    b) return the key with the max value"""  
    v = list(d.values())
    k = list(d.keys())
    if idx == 0:
        return k[v.index(max(v,key=itemgetter(idx)))]
    else:
        return k[v.index(min(v,key=itemgetter(idx)))]
        
def get_best(method):
    if method == "AC":
        with open('network_params_ac.pickle', 'rb+') as handle:
            results_dict = pickle.load(handle)
    else:
        with open('network_params_reinforce.pickle', 'rb+') as handle:
            results_dict = pickle.load(handle)
    return keywithmaxval(results_dict, 0)

def run_experiments(method, total_episodes = 1000, learning_rate = None, future_discount = 1, estimation_depth = 500, gradient_method = 'both', hidden_shape = None, hidden_shape_actor = None, hidden_shape_critic = None, hidden_layers = None, hidden_layers_actor = None, hidden_layers_critic = None, tune = 0, repetitions = 5, silent=True):
    if int(tune) > 0:
        results_dict = dict()
        lr = [.5, 1e-1, 1e-2, 1e-3]
        # First stochastically tune network architecture
        if method == "AC":
            if not exists('network_params_ac.pickle'):
                Path('network_params_ac.pickle').touch()
            else:
                with open('network_params_ac.pickle', 'rb+') as handle:
                    results_dict = pickle.load(handle)
        if method == "REINFORCE":
            if not exists('network_params_reinforce.pickle'):
                Path('network_params_reinforce.pickle').touch()
            else:
                with open('network_params_reinforce.pickle', 'rb+') as handle:
                    results_dict = pickle.load(handle)
        prev_len = len(results_dict)
        for i in tqdm(range(int(tune))):
            if method == "AC":
                ha = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                hc = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                lrate = random.choice(lr)
                dummy_key = {'layers_actor' : len(ha), 'layers_critic' : len(hc), 'nodes_actor' : tuple(ha), 'nodes_critic' : tuple(hc), 'lr': lrate}
                key = frozenset(dummy_key.items())
                while key in results_dict:#results_dict.has_key(key):
                    ha = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                    hc = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                    lrate = random.choice(lr)
                    dummy_key = {'layers_actor' : len(ha), 'layers_critic' : len(hc), 'nodes_actor' : tuple(ha), 'nodes_critic' : tuple(hc), 'lr' : lrate}
                    key = frozenset(dummy_key.items())
                results1 = []
                results2 = []
                for _ in range(int(repetitions)):
                    result = AC(total_episodes, estimation_depth, lrate, gradient_method, ha, hc, silent)
                    results1.append(sum(result[0])/len(result[0]))
                    results2.append(result[1])
                results_dict[key] = (sum(results1)/len(results1),sum(results2)/len(results2))
            if method == "REINFORCE":
                hl = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                lrate = random.choice(lr)
                dummy_key = {'layers' : len(hl), 'nodes' : tuple(hl), 'lr': lrate}
                key = frozenset(dummy_key.items())
                while key in results_dict:
                    hl = [int(np.random.randint(1, 8, 1))*16 for _ in range(np.random.randint(1, 5))]
                    lrate = random.choice(lr)
                    dummy_key = {'layers' : len(hl), 'nodes' : tuple(hl), 'lr': lrate}
                    key = frozenset(dummy_key.items())
                results1 = []
                results2 = []
                for _ in range(int(repetitions)):
                    result = Cartpole(total_episodes, lrate, future_discount, hl, silent)
                    results1.append(sum(result[0])/len(result[0]))
                    results2.append(result[1])
                results_dict[key] = (sum(results1)/len(results1),sum(results2)/len(results2))
            # If exhausted, stop
            if prev_len >= len(results_dict):
                break
            if method == "AC":
                with open('network_params_ac.pickle', 'wb') as handle:
                    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if method == "REINFORCE":
                with open('network_params_reinforce.pickle', 'wb') as handle:
                    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:        
        settings = dict(get_best(method))
        #settings = {tuple(map(tuple, k)): v for k, v in frozen_settings.items()}
        if learning_rate is None:
            learning_rate = settings['lr']
            
        if method == "AC":
            if hidden_shape_actor is None:
                hidden_shape_actor = settings['nodes_actor']
            if hidden_shape_critic is None:
                hidden_shape_critic = settings['nodes_critic']
            if hidden_layers_actor is None and hidden_layers_critic is None:
                score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic, silent)
            else:
                score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, [hidden_shape_actor for _ in range(hidden_layers_actor)], [hidden_shape_critic for _ in range(hidden_layers_critic)], silent)
        if method == "REINFORCE":
            if hidden_shape is None:
                hidden_shape = settings['nodes']
            if hidden_layers is None:
                score = Cartpole(total_episodes, learning_rate, future_discount, hidden_shape, silent)
            else:
                score = Cartpole(total_episodes, learning_rate, future_discount, [hidden_shape for _ in range(hidden_layers)], silent)
        plot_results(total_episodes, score)

def best_settings(method):
    if method == "AC":
        pass
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
    parser.add_argument('-tune', help='Tune parameters, where \"tune\" is the amount of dependent parameters to tune. Put 0 for no tuning', dest = 'tune', required = False)
    parser.add_argument('-reps', help='The amount of repetitions to average results over', dest = 'reps', required = False)
    parser.add_argument('-silent', help='Print score', dest = 'silent', required = False)
    args = parser.parse_args()
    kwargs = dict(total_episodes=args.episodes,learning_rate=args.learning_rate,future_discount=args.discount,estimation_depth=args.estimation,
    gradient_method=args.gradient,hidden_shape=args.hidden,hidden_shape_actor=args.hidden_actor,hidden_shape_critic=args.hidden_critic,
    hidden_layers=args.layers,hidden_layers_actor=args.layers_actor,hidden_layers_critic=args.layers_critic,tune=args.tune,repetitions=args.reps, silent=int(args.silent))
    
    run_experiments(args.method, **{k: v for k, v in kwargs.items() if v is not None})

if __name__ == "__main__":
    main()
