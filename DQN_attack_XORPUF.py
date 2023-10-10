# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:57:16 2023

@author: Mieszko Ferens

Script to run an experiment for modelling an Arbiter or XOR PUF using Double
DQN. Requires ChainerRL v0.8.0.
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
    parser.add_argument("--k", type=int, default=1,
                        help="Number of parallel APUFs in the XOR PUF.")
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
    env = PUF_env(args.n_bits, args.k, seed=args.seed)
    env.reset(args.train_data, args.test_data)
    
    # --- Create agent
    
    # Q function
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        env.obs_size, env.n_actions, args.hidden_layer_size,
        args.n_hidden_layers)

    # Explorer
    random_selector = lambda : np.random.choice(env.n_actions)
    explorer = explorers.ConstantEpsilonGreedy(
        args.epsilon, random_action_func=random_selector)

    # Optimizer
    optimizer = optimizers.Adam(eps=1e-3)
    optimizer.setup(q_func)

    # Replay buffer
    r_buffer = replay_buffer.ReplayBuffer(capacity=args.replay_buffer_size)

    # Input converter
    phi = lambda x : x.astype(np.float32, copy=False)

    # Agent
    agent = agents.DoubleDQN(
        q_func, optimizer, r_buffer, args.gamma, explorer,
        replay_start_size=args.replay_start_size,
        minibatch_size=args.batch_size, update_interval=args.update_interval,
        target_update_interval=args.target_update_interval, phi=phi)
    
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
            
            # Act on the challenge and train
            action = agent.act_and_train(obs, reward)
            
            # Check result
            _ , reward, done = env.step(action)
        
        # Train on episode end
        agent.stop_episode_and_train(obs, reward, done)
        
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
        action = agent.act(obs)
        
        # Check prediction
        _ , reward, _ = env.step(action)
        total_reward += reward

    print("Testing complete")

    # Calculate accuracy
    accuracy = (total_reward/args.test_data)

    print("Final accuracy: " + str(accuracy))
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
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
    filepath = Path(args.outdir + "out_DQN_" + str(args.k) + "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    
    
if(__name__ == "__main__"):
    main()
    
    