# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:21:08 2023

@author: Mieszko Ferens

Script to run an experiment for modelling an XOR Arbiter PUF using LR.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
import pypuf.attack

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def main():
    
    # Set-up logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./Results/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of parallel APUFs in the XOR PUF.")
    parser.add_argument("--train-data", type=int, default=50000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=10000,
                        help="Number of testing data samples for the model.")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = XORArbiterPUF(args.n_bits, args.k, args.seed)
    
    # Generate the challenges
    challenges = random_inputs(
        args.n_bits, args.train_data + args.test_data, args.seed)
    
    # Get responses
    responses = puf.eval(challenges)
    
    # Split the data into training and testing
    train = args.train_data
    test = train + args.test_data
    
    # Prepare the data for training and testing
    train_crps = ChallengeResponseSet(
        args.n_bits, np.array(challenges[:train], dtype=np.int8),
        np.array(responses[:train], dtype=np.float64))
    test_x = challenges[train:test]
    test_y = np.expand_dims(0.5 - 0.5*responses[train:test], axis=1)
    
    # Use an LR as a predictor
    model = pypuf.attack.LRAttack2021(
        train_crps, seed=args.seed, k=args.k, epochs=100, lr=.001, bs=1000,
        stop_validation_accuracy=.97)
    
    # Train the model
    model.fit()

    # Test the model
    pred_y = model._model.eval(test_x)
    pred_y = pred_y.reshape(len(pred_y), 1)
    
    # Calculate accuracy
    accuracy = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
    print("---\n" +
          "Accuracy in the testing data: " + str(accuracy*100) + "%")
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
                         "train_data": [args.train_data],
                         "test_data": [args.test_data],
                         "accuracy": [accuracy]})
    filepath = Path(args.outdir + "out_LR_" + str(args.k) + "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()

