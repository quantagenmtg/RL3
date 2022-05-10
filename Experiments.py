# %%
import numpy as np
import torch
import gym
import random
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import seaborn as sns
import pandas as pd
from torch import FloatTensor,Tensor,LongTensor
from Policy import REINFORCE, ActorCritic
from os.path import exists
from pathlib import Path
from tqdm import tqdm
from operator import itemgetter
import scipy.stats as st

# # REINFORCE
# We define a function to run REINFORCE algorithm on.

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
        learned_at: if model converged the episode it converged
        loss: Loss per episode in a list
    """
    
    
    scores = []
    loss = []
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
        loss.append(agent.update(rewards,log_probs))

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
    for _ in range(total_episodes-len(scores)):
        scores.append(scores[-1])
    return scores, learned_at, loss

    
# # Actor-Critic

# %%
def AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic, silent = True):
    """
    Tries to solve Cartpole-v1 usinf the Actor Critic algorithm.

    Args:
        total_episodes: How many times the environment resets
        learning_rate: For optimizer
        hidden_shape: List of integers. [16,16] would give two hidden layers (linear with PReLU activation) with both 16 nodes in the policy model
        estimation_depth: depth that critic estimates the environment
        gradient_method: Determines whether to use bootstrapping, baseline subtraction, or both
    Returns:
        scores: Score per episode in a list
        learned_at: if model converged the episode it converged
        loss: Loss per episode in a list
    """
    scores = []
    loss = []
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
        loss.append(agent.update(rewards,log_probs, states))

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
    for _ in range(total_episodes-len(scores)):
        scores.append(scores[-1])
    return scores, learned_at, loss

    

# Plots results of default experients
def plot_results(scores, total_episodes, method, plot_grads = False):
    fixed_scores = []
    # Cleaning up lists
    for score in scores:
        lst = [sum(score)/len(score) if np.isnan(x) else x for x in score]
        for _ in range(total_episodes-len(lst)):
            lst.append(score[-1])
        fixed_scores.append(lst)
    # For error bars around plots
    error = st.t.interval(alpha=0.95, df=len(fixed_scores)-1, loc=np.mean(fixed_scores, axis=0), scale=st.sem(fixed_scores)) 
    plt.plot(np.arange(1,total_episodes+1), np.mean(fixed_scores, axis=0), 'b-')
    plt.fill_between(np.arange(1,total_episodes+1), error[0], error[1], color='b', alpha=0.2)
    plt.xlabel("Episode")
    if plot_grads:
        plt.ylabel("Loss")
    else:
        plt.ylabel("score")
    plt.title(method)
    plt.show()

# Plots ablation results
def plot_ablation_results(total_episodes, scores, plot_grads = False):
    fixed_scores1 = []
    for score in scores[0]:
        lst = [sum(score)/len(score) if np.isnan(x) else x for x in score]
        for _ in range(total_episodes-len(lst)):
            lst.append(score[-1])
        fixed_scores1.append(lst)
    fixed_scores2 = []
    for score in scores[1]:
        lst = [sum(score)/len(score) if np.isnan(x) else x for x in score]
        for _ in range(total_episodes-len(lst)):
            lst.append(score[-1])
        fixed_scores2.append(lst)
    fixed_scores3 = []
    for score in scores[2]:
        lst = [sum(score)/len(score) if np.isnan(x) else x for x in score]
        for _ in range(total_episodes-len(lst)):
            lst.append(score[-1])
        fixed_scores3.append(lst)
    error1 = st.t.interval(alpha=0.95, df=len(fixed_scores1)-1, loc=np.mean(fixed_scores1, axis=0), scale=st.sem(fixed_scores1)) 
    error2 = st.t.interval(alpha=0.95, df=len(fixed_scores2)-1, loc=np.mean(fixed_scores2, axis=0), scale=st.sem(fixed_scores2)) 
    error3 = st.t.interval(alpha=0.95, df=len(fixed_scores3)-1, loc=np.mean(fixed_scores3, axis=0), scale=st.sem(fixed_scores3)) 
    plt.plot(np.arange(1,total_episodes+1), np.mean(fixed_scores1, axis=0), 'b-', label='nsteps')
    plt.fill_between(np.arange(1,total_episodes+1), error1[0], error1[1], color='b', alpha=0.2)
    plt.plot(np.arange(1,total_episodes+1), np.mean(fixed_scores2, axis=0), 'r-', label='baseline')
    plt.fill_between(np.arange(1,total_episodes+1), error2[0], error2[1], color='r', alpha=0.2)
    plt.plot(np.arange(1,total_episodes+1), np.mean(fixed_scores3, axis=0), 'g-', label='both')
    plt.fill_between(np.arange(1,total_episodes+1), error3[0], error3[1], color='g', alpha=0.2)
    plt.xlabel("Episode")
    if plot_grads:
        plt.ylabel("Loss")
    else:
        plt.ylabel("score")
    plt.legend(loc="upper left")
    plt.show()

# Plots policy comparison
def plot_compare_results(total_episodes, scores):
    error1 = st.t.interval(alpha=0.95, df=len(scores[0])-1, loc=np.mean(scores[0], axis=0), scale=st.sem(scores[0])) 
    error2 = st.t.interval(alpha=0.95, df=len(scores[1])-1, loc=np.mean(scores[1], axis=0), scale=st.sem(scores[1])) 
    plt.plot(np.arange(1,total_episodes+1), np.mean(np.asarray(scores[0]), axis=0), 'b-', label='AC')
    plt.fill_between(np.arange(1,total_episodes+1), error1[0], error1[1], color='b', alpha=0.2)
    plt.plot(np.arange(1,total_episodes+1), np.mean(np.asarray(scores[1]), axis=0), 'r-', label='REINFORCE')
    plt.fill_between(np.arange(1,total_episodes+1), error2[0], error2[1], color='r', alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    plt.show()

def keywithmaxval(d, idx):
    """ a) create a list of the dict's keys and values; 
    b) return the key with the max value"""  
    v = list(d.values())
    k = list(d.keys())
    if idx == 0:
        return k[v.index(max(v,key=itemgetter(idx)))]
    else:
        return k[v.index(min(v,key=itemgetter(idx)))]

# Gets best parameter settings from tuning dict   
def get_best(method):
    if method == "AC":
        with open('network_params_ac.pickle', 'rb+') as handle:
            network_dict = pickle.load(handle)
        with open('params_ac.pickle', 'rb+') as handle:
            param_dict = pickle.load(handle)
    else:
        with open('network_params_reinforce.pickle', 'rb+') as handle:
            network_dict = pickle.load(handle)
        with open('params_reinforce.pickle', 'rb+') as handle:
            param_dict = pickle.load(handle)
    return keywithmaxval(network_dict, 0), keywithmaxval(param_dict, 0)


def run_experiments(method, total_episodes = 1000, learning_rate = None, future_discount = 1, estimation_depth = 500, gradient_method = 'both', hidden_shape = None, hidden_shape_actor = None, hidden_shape_critic = None, hidden_layers = None, hidden_layers_actor = None, hidden_layers_critic = None, tune = 0, repetitions = 5, silent=True, ablation=False, compare=False, plot_grads=False):
    '''
        DISCLAIMER: It would be neater to divide this method into multiple ones but it works really well and otherwise we need to pass the 1000 function arguments to the other methods
        
        Executes experiments according to arguments and plots results (either loss or score)

        Args:
            method: Which policy to test ("AC" or "REINFORCE)
            total_episodes: How many times the environment resets
            learning_rate: For optimizer
            future_reward_discount_factor: future rewards are dicounted
            estimation_depth: depth that critic estimates the environment
            gradient_method: Determines whether to use bootstrapping, baseline subtraction, or both
            shapes and nodes: network settings
            tune: integer, if > 0, it will tune the model for that number of settings. it also does not plot results
            repetitions: integer, number of times to test the model
            silent: boolean, if False it prints score per episode
            ablation: boolean, if True it performs the ablation experiments
            compare: boolean, if True it compares REINFORCE performance with AC
            plot_grads: Plot Loss as metric instead of Score per episode
    '''
    if silent is not None:
        silent = int(silent)
    if repetitions is not None:
        repetitions = int(repetitions)
    if ablation is not None:
        ablation = int(ablation)
    if compare is not None:
        compare = int(compare)
    total_episodes = int(total_episodes)
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

            network_settings, _ = get_best(method)
            network_settings = dict(network_settings)
            results_dict = dict()
            if method == "AC": 
                gms = ['nstep', 'baseline', 'both']
                depths = [1, 50, 100, 500]
                for gm in tqdm(gms):
                    for depth in tqdm(depths, leave=False):
                        dummy_key = {'gms' : gm, 'depth' : depth}
                        key = frozenset(dummy_key.items())
                        results1 = []
                        results2 = []
                        for _ in range(int(repetitions)):
                            result = AC(total_episodes, depth, network_settings['lr'], gm, network_settings['nodes_actor'], network_settings['nodes_critic'], silent)
                            results1.append(sum(result[0])/len(result[0]))
                            results2.append(result[1])
                        results_dict[key] = (sum(results1)/len(results1),sum(results2)/len(results2))
            if method == "REINFORCE": 
                discounts = [.95, .99, .995, ]
                for d in tqdm(discounts):
                    dummy_key = {'discount': d}
                    key = frozenset(dummy_key.items())
                    results1 = []
                    results2 = []
                    for _ in range(int(repetitions)):
                        result = Cartpole(total_episodes, network_settings['lr'], d, network_settings['nodes'], silent)
                        results1.append(sum(result[0])/len(result[0]))
                        results2.append(result[1])
                    results_dict[key] = (sum(results1)/len(results1),sum(results2)/len(results2))
            if method == "AC":
                with open('params_ac.pickle', 'wb') as handle:
                    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if method == "REINFORCE":
                with open('params_reinforce.pickle', 'wb') as handle:
                    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif ablation:
        ablation_score = []
        ablation_loss = []
        scores = []
        losses = []
        network_settings, param_settings = get_best(method)
        network_settings = dict(network_settings)
        param_settings = dict(param_settings)
        if learning_rate is None:
            learning_rate = network_settings['lr']
        for gm in ['nstep', 'baseline', 'both']:
            gradient_method =gm
            for _ in range(repetitions):
                if method == "AC":
                    if estimation_depth is None:
                        estimation_depth = param_settings['depth']
                    if hidden_shape_actor is None:
                        hidden_shape_actor = network_settings['nodes_actor']
                    if hidden_shape_critic is None:
                        hidden_shape_critic = network_settings['nodes_critic']
                    if hidden_layers_actor is None and hidden_layers_critic is None:
                        score = AC(total_episodes, estimation_depth, learning_rate, gm, hidden_shape_actor, hidden_shape_critic, silent)
                    else:
                        score = AC(total_episodes, estimation_depth, learning_rate, gm, [hidden_shape_actor for _ in range(hidden_layers_actor)], [hidden_shape_critic for _ in range(hidden_layers_critic)], silent)
                scores.append(score[0])
                losses.append(score[2])
            ablation_score.append(scores)
            ablation_loss.append(losses)
            losses = []
            scores = []
        if plot_grads:
            plot_ablation_results(total_episodes, ablation_loss, plot_grads)
        else:
            plot_ablation_results(total_episodes, ablation_score)
    elif compare:
        scores = []
        method_scores = []
        for m in ['AC', 'REINFORCE']:
            network_settings, param_settings = get_best(m)
            network_settings = dict(network_settings)
            param_settings = dict(param_settings)
            if learning_rate is None:
                learning_rate = network_settings['lr']
            for _ in range(repetitions):
                if m == "AC":
                    if gradient_method is None:
                        gradient_method = param_settings['gms']
                    if estimation_depth is None:
                        estimation_depth = param_settings['depth']
                    if hidden_shape_actor is None:
                        hidden_shape_actor = network_settings['nodes_actor']
                    if hidden_shape_critic is None:
                        hidden_shape_critic = network_settings['nodes_critic']
                    if hidden_layers_actor is None and hidden_layers_critic is None:
                        score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic, silent)
                    else:
                        score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, [hidden_shape_actor for _ in range(hidden_layers_actor)], [hidden_shape_critic for _ in range(hidden_layers_critic)], silent)
                if m == "REINFORCE":
                    if future_discount is None:
                        future_discount = param_settings['discount']
                    if hidden_shape is None:
                        hidden_shape = network_settings['nodes']
                    if hidden_layers is None:
                        score = Cartpole(total_episodes, learning_rate, future_discount, hidden_shape, silent)
                    else:
                        score = Cartpole(total_episodes, learning_rate, future_discount, [hidden_shape for _ in range(hidden_layers)], silent)
                scores.append(score[0])
            method_scores.append(scores)
            scores = []
            
        plot_compare_results(total_episodes, method_scores)
    else:        
        scores = []
        losses = []
        network_settings, param_settings = get_best(method)
        network_settings = dict(network_settings)
        param_settings = dict(param_settings)
        if learning_rate is None:
            learning_rate = network_settings['lr']
        for _ in range(repetitions):
            if method == "AC":
                if gradient_method is None:
                    gradient_method = param_settings['gms']
                if estimation_depth is None:
                    estimation_depth = param_settings['depth']
                if hidden_shape_actor is None:
                    hidden_shape_actor = network_settings['nodes_actor']
                if hidden_shape_critic is None:
                    hidden_shape_critic = network_settings['nodes_critic']
                if hidden_layers_actor is None and hidden_layers_critic is None:
                    score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, hidden_shape_actor, hidden_shape_critic, silent)
                else:
                    score = AC(total_episodes, estimation_depth, learning_rate, gradient_method, [hidden_shape_actor for _ in range(hidden_layers_actor)], [hidden_shape_critic for _ in range(hidden_layers_critic)], silent)
            if method == "REINFORCE":
                if future_discount is None:
                    future_discount = param_settings['discount']
                if hidden_shape is None:
                    hidden_shape = network_settings['nodes']
                if hidden_layers is None:
                    score = Cartpole(total_episodes, learning_rate, future_discount, hidden_shape, silent)
                else:
                    score = Cartpole(total_episodes, learning_rate, future_discount, [hidden_shape for _ in range(hidden_layers)], silent)
            scores.append(score[0])
            losses.append(score[2])
        if plot_grads:
            plot_results(losses, total_episodes, method, plot_grads)
        else:
            plot_results(scores, total_episodes, method)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='Method', dest = 'method', required = True)
    parser.add_argument('-eps', help='Total episodes', dest = 'episodes', required=False)
    parser.add_argument('-lr', help='Learning rate', dest = 'learning_rate', required=False)
    parser.add_argument('-d', help='Future reward discount factor (REINFORCE)', dest = 'discount', required=False)
    parser.add_argument('-est', help='Estimation depth (AC)', dest = 'estimation', required=False)
    parser.add_argument('-grad', help='Gradient method (AC): \"nstep\", \"baseline\", or \"both\"', dest = 'gradient', required=False)
    parser.add_argument('-hs', help='Hidden shape (REINFORCE)', dest = 'hidden', required=False)
    parser.add_argument('-ha', help='Hidden shape actor (AC)', dest = 'hidden_actor', required=False)
    parser.add_argument('-hc', help='Hidden shape critic (AC))', dest = 'hidden_critic', required=False)
    parser.add_argument('-l', help='Hidden layers (REINFORCE)', dest = 'layers', required=False)
    parser.add_argument('-la', help='Hidden layers actor (AC)', dest = 'layers_actor', required=False)
    parser.add_argument('-lc', help='Hidden layers critic (AC)', dest = 'layers_critic', required=False)
    parser.add_argument('-tune', help='Tune parameters, where \"tune\" is the amount of dependent parameters to tune. Put 0 for no tuning', dest = 'tune', required = False)
    parser.add_argument('-reps', help='The amount of repetitions to average results over', dest = 'reps', required = False)
    parser.add_argument('-silent', help='Print score', dest = 'silent', required = False)
    parser.add_argument('-ablation', help='Plot ablation experiments', dest = 'ablation', required = False)
    parser.add_argument('-compare', help='Plot comparisons', dest = 'compare', required = False)
    parser.add_argument('-variance', help='Plot loss instead of score', dest = 'plot_grads', required = False)
    args = parser.parse_args()
    kwargs = dict(total_episodes=args.episodes,learning_rate=args.learning_rate,future_discount=args.discount,estimation_depth=args.estimation,
    gradient_method=args.gradient,hidden_shape=args.hidden,hidden_shape_actor=args.hidden_actor,hidden_shape_critic=args.hidden_critic,
    hidden_layers=args.layers,hidden_layers_actor=args.layers_actor,hidden_layers_critic=args.layers_critic,tune=args.tune,repetitions=args.reps, silent=args.silent,
    ablation=args.ablation,compare=args.compare,plot_grads=args.plot_grads)
    
    if args.method != "REINFORCE" and args.method != "AC":
        print("Please provide a valid method (\"REINFORCE\" or \"AC\")")
    else:
        run_experiments(args.method, **{k: v for k, v in kwargs.items() if v is not None})

if __name__ == "__main__":
    main()

