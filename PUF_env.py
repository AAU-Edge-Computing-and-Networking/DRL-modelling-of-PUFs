# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:24:34 2022

@author: Mieszko Ferens
"""

import numpy as np
from pypuf.simulation import XORArbiterPUF, InterposePUF
from pypuf.io import random_inputs

class PUF_env():
    
    """
    Class which implements an episodic training environment for RL agents that
    uses the pypuf library to generate a puf instance
    
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
    
    def __init__(self, challenge_len, k, puf="XORPUF", seed=None):
        
        # Create PUF
        if(puf == "XORPUF"):
            self.puf = XORArbiterPUF(n=challenge_len, k=k, seed=seed)
        elif(puf == "IPUF"):
            self.puf = InterposePUF(n=challenge_len, k_down=k, seed=seed)
        
        # Create RNG
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        
        # Challenges
        self.c = np.array([])
        # Responses
        self.r = np.array([])
        
        # Current CRP
        self.CRP = 0
        
        # Observation
        self.obs_size = challenge_len
        self.obs = np.zeros(self.obs_size)
        # Action space
        self.n_actions = 2 # 0 or 1
        self.action = 0
    
    def step(self, action, episodic=True):
        
        """
        Check if an action matches the correct response of the PUF and end the
        episode if training is episodic and action is correct
        
        Arguments:
            action (int): The action to test
            episodic (boolean, default=True): Whether to use episodic or
            continuous training
        
        Returns:
            Tuple that contains the observation (used challenge), reward and
            boolean of whether the episode is over
        """
        
        # Check if action matches the response to the challenge
        if(action == self.r[self.CRP]):
            reward = +1
            done = True
        else:
            reward = 0
            done = False
        
        if(not episodic):
            done = False
        
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
        self.c = random_inputs(self.obs_size, train_samples + test_samples,
                               seed=self.seed)
        
        # Get responses ((0,1) values)
        self.r = np.int8(0.5 - 0.5*self.puf.eval(self.c))
        
        # Split data into training and testing sets randomly (with indices)
        self.train = train_samples
        self.test = test_samples
    
    def set_CRP(self, index):
        
        """
        Method that sets the current CRP used in the environment and calculates
        the observation (transformed challenge)
        
        Arguments:
            index (int): The index of the CRP
        
        Returns:
            Observation
        """
        
        # Set CRP index
        self.CRP = index
        
        # Transform challenge
        self.obs = np.cumprod(
            np.fliplr(np.array([self.c[index]])), axis=1, dtype=np.int8)[0]
        
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

