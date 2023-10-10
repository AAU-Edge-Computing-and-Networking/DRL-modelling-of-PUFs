# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:19:32 2023

@author: Mieszko Ferens

Script to run an experiment for modelling an Interpose PUF using collaborative
Double DQN. Requires ChainerRL v0.8.0.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from chainer import optimizers
from chainerrl import agents, q_functions, explorers, replay_buffer
from PUF_env import PUF_env

def main():
    
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./Results/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k-down", type=int, default=1,
                        help="Number of parallel APUFs in the XOR PUF of the" +
                        " lower layer of the IPUF.")
    parser.add_argument("--train-data", type=int, default=10000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=1000,
                        help="Number of testing data samples for the model.")
    parser.add_argument("--hidden-layer-size", type=int, default=16,
                        help="Size of all hidden layers of the nueral network.")
    parser.add_argument("--n-hidden-layers", type=int, default=2,
                        help="Number of hidden layers of the neural network.")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Probability of exploratory (random) action.")
    parser.add_argument("--replay-buffer-size", type=int, default=25000,
                        help="Experience replay buffer size.")
    parser.add_argument("--replay-start-size", type=int, default=5000,
                        help="Min replay buffer size to use experience replay.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Experience replay batch size.")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Discount factor.")
    parser.add_argument("--update-interval", type=int, default=1,
                        help="Number of steps for updating main model.")
    parser.add_argument("--target-update-interval", type=int, default=500,
                        help="Number of steps for updating target model.")
    parser.add_argument("--max-episode-len", type=int, default=2,
                        help="Max number of steps per training episode.")
    args = parser.parse_args()
    
    # --- Create PUF environment
    env = PUF_env(args.n_bits, args.k_down, puf="IPUF", seed=args.seed)
    env.reset(args.train_data, args.test_data)
    
    # --- Create agent for upper layer (always APUF)
    
    # Q function
    q_func_U = q_functions.FCStateQFunctionWithDiscreteAction(
        env.obs_size, env.n_actions, n_hidden_channels=16, n_hidden_layers=2)

    # Explorer
    random_selector_U = lambda : np.random.choice(env.n_actions)
    explorer_U = explorers.ConstantEpsilonGreedy(
        epsilon=0.2, random_action_func=random_selector_U)

    # Optimizer
    optimizer_U = optimizers.Adam(eps=1e-3)
    optimizer_U.setup(q_func_U)

    # Replay buffer
    r_buffer_U = replay_buffer.ReplayBuffer(capacity=25000)

    # Input converter
    phi_U = lambda x : x.astype(np.float32, copy=False)

    # Agent
    agent_U = agents.DoubleDQN(
        q_func_U, optimizer_U, r_buffer_U, gamma=0.1, explorer=explorer_U,
        replay_start_size=5000, minibatch_size=32, update_interval=1,
        target_update_interval=500, phi=phi_U)
    
    # --- Create agent for lower layer
    
    # Q function
    q_func_L = q_functions.FCStateQFunctionWithDiscreteAction(
        env.obs_size + 1, env.n_actions, args.hidden_layer_size,
        args.n_hidden_layers)

    # Explorer
    random_selector_L = lambda : np.random.choice(env.n_actions)
    explorer_L = explorers.ConstantEpsilonGreedy(
        args.epsilon, random_action_func=random_selector_L)

    # Optimizer
    optimizer_L = optimizers.Adam(eps=1e-3)
    optimizer_L.setup(q_func_L)

    # Replay buffer
    r_buffer_L = replay_buffer.ReplayBuffer(capacity=args.replay_buffer_size)

    # Input converter
    phi_L = lambda x : x.astype(np.float32, copy=False)

    # Agent
    agent_L = agents.DoubleDQN(
        q_func_L, optimizer_L, r_buffer_L, args.gamma, explorer_L,
        replay_start_size=args.replay_start_size,
        minibatch_size=args.batch_size, update_interval=args.update_interval,
        target_update_interval=args.target_update_interval, phi=phi_L)
    
    # --- Train the agent
    
    print("Training...")

    steps = 0 # Total number of training steps
    for i in range(args.train_data):
        
        # Set CRP and get challenge
        obs = env.set_CRP(i)
        reward = 0
        done = False
        
        t = 0 # Time step
        while(t < args.max_episode_len and not done):
            
            # Increment total and episodic time steps
            steps += 1
            t += 1
            
            # Predict respose from upper layer
            action_U = agent_U.act_and_train(obs, reward)
            
            # Generate predicted challenge for lower layer
            c = np.insert(env.c[env.CRP], 32, 1 - (2*action_U))
            obs_L = np.cumprod(
                np.fliplr(np.array([c])), axis=1, dtype=np.int8)[0]
            
            # Predict response from lower layer
            action_L = agent_L.act_and_train(obs_L, reward)
            
            # Check result
            _ , reward, done = env.step(action_L)
        
        # Train on episode end
        agent_U.stop_episode_and_train(obs, reward, done)
        agent_L.stop_episode_and_train(obs_L, reward, done)
        
        if(not (i+1) % 1000):
            print("Episode: " + str(i+1))
            print("Steps: " + str(steps))

    print("Training complete")
    
    # --- Test the agent ---

    print("Testing...")

    total_reward = 0
    for i in range(args.train_data, args.train_data + args.test_data):
        
        # Set CRP and get challenge
        obs = env.set_CRP(i)
        
        # Predict response
        action_U = agent_U.act(obs)
        c = np.insert(env.c[env.CRP], 32, 1 - (2*action_U))
        obs_L = np.cumprod(
            np.fliplr(np.array([c])), axis=1, dtype=np.int8)[0]
        action_L = agent_L.act(obs_L)
        
        # Check prediction
        _ , reward, _ = env.step(action_L)
        total_reward += reward

    print("Testing complete")

    # Calculate accuracy
    accuracy = (total_reward/args.test_data)

    print("Final accuracy: " + str(accuracy))
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k_down": [args.k_down],
                         "train_data": [args.train_data],
                         "test_data": [args.test_data],
                         "gamma": [args.gamma],
                         "epsilon": [args.epsilon],
                         "n_hidden_layers": [args.n_hidden_layers],
                         "hidden_layer_size": [args.hidden_layer_size],
                         "replay_buffer_size": [args.replay_buffer_size],
                         "batch_size": [args.batch_size],
                         "update_interval": [args.update_interval],
                         "target_update_interval": [args.target_update_interval],
                         "max_episode_len": [args.max_episode_len],
                         "accuracy": [accuracy]})
    filepath = Path(args.outdir + "out_DQN_1" + str(args.k_down) + "IPUF.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    
    
if(__name__ == "__main__"):
    main()
    
    