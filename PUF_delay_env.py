# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:05:25 2023

@author: Mieszko Ferens
"""

import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs

class PUF_delay_env():
    
    """
    Class which implements an episodic training environment for RL agents that
    uses the pypuf library to generate a puf instance. The agent is tasked to
    predict the delay difference at each stage of the PUF, not just the final
    response.
    
    Reference:
        N. Wisiol, C. Gräbnitz, C. Mühl, B. Zengin, T. Soroceanu, N. Pirnay,
        K. T. Mursi, & A. Baliuka, "pypuf: Cryptanalysis of Physically
        Unclonable Functions" (Version 2, June 2021). Zenodo.
        https://doi.org/10.5281/zenodo.3901410
    
    Arguments:
        challenge_len (int): Challenge length
        seed (int, default=None): The seed to pass to the RNG
        + Requires the pypuf library to be installed (>=2.2.0)
    
    Returns:
        PUF environment object
    """
    
    def __init__(self, challenge_len, k, granularity, seed=None):
        
        # Create PUF
        self.puf = XORArbiterPUF(n=challenge_len, k=k, seed=seed)
        
        # Create RNG
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        
        # Challenges
        self.c = np.array([])
        self.c_len = challenge_len
        # Responses
        self.r = np.array([])
        
        # Current CRP
        self.CRP = 0
        
        # Current stage of the PUF
        self.stage = 0
        
        # Observation
        self.obs_size = 2*self.c_len + 2
        self.obs = np.zeros(self.obs_size)
        # Action space
        self.n_actions = granularity
        self.action = 0
    
    def step(self, action):
        
        """
        Check if an action matches the correct response of the PUF and end the
        episode if training is episodic and action is correct
        
        Arguments:
            action (int): The action to test
        
        Returns:
            Tuple that contains the observation (used challenge), reward and
            boolean of whether the episode is over
        """
        
        # Default reward value (only final stage prediction can provide reward)
        reward = 0
        
        # Update the current stage
        self.stage += 1
        
        # Update the currently predicted delay difference
        if(action >= (self.n_actions - 1) / 2):
            self.obs[-1] += 2*(action - ((self.n_actions - 1) / 2))
        
        # If action predicted delay on final stage
        if(self.stage == self.c_len - 1):
            done = True
            response = 0
            if(action >= (self.n_actions - 1) / 2):
                response = 1
            if(response == self.r[self.CRP]):
                reward = +1
        else: # Set next stage
            done = False
            self.obs[self.c_len+self.stage : self.c_len+self.stage+2] = [0, 1]
            self.obs[-2] = self.c[self.CRP][self.stage]
        
        return self.obs, reward, done
    
    def reset(self, train_samples=10000, test_samples=1000):
        
        """
        Reset the environment by generating all CRPs and splitting them into
        training and testing datasets
        
        Arguments:
            train_samples (int, default=10000): Number of training CRP samples
            for the model
            test_samples (int, default=1000): Number of testing CRP samples for
            the model
        """
        
        # Get challenges ((-1,1) values)
        self.c = random_inputs(self.c_len, train_samples + test_samples,
                               seed=self.seed)
        
        # Get responses ((0,1) values)
        self.r = np.int8(0.5 - 0.5*self.puf.eval(self.c))
        
        # Split data into training and testing sets randomly (with indices)
        self.train = train_samples
        self.test = test_samples
    
    def set_CRP(self, index):
        
        """
        Method that sets the current CRP used in the environment and calculates
        the initial observation (including the transformed challenge)
        
        Arguments:
            index (int): The index of the CRP
        
        Returns:
            Observation
        """
        
        # Set CRP index
        self.CRP = index
        
        # Set stage to first
        self.stage = 0
        
        # Transform challenge and generate initial observation for episode
        self.obs = np.append(
            np.cumprod(np.fliplr(np.array([self.c[self.CRP]])), axis=1,
                       dtype=np.int8)[0],
            np.zeros(self.c_len + 2))
        self.obs[self.c_len] = 1
        self.obs[-2] = self.c[self.CRP][self.stage]
        self.obs[-1] = 0
        
        return self.obs
    
    def next_CRP(self):
        """
        Method that sets the current CRP used in the environment to the next
        one in the list, and calculates the observation (transformed challenge)
        
        Returns:
            Observation
        """
        
        if(self.CRP+1 == self.train):
            print("Warning: Steping out of training CRPs and into testing CRPs")
        
        return self.set_CRP(self.CRP+1)

