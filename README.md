# DRL-modelling-of-PUFs

Code to run Deep Reinforcement Learning (DRL) modelling attacks against Delay-based PUFs.

The available PUFs are:
 - Arbiter PUF
 - k-XOR PUF
 - Interpose PUF

The DRL algorithm used is Deep Q-Network (DQN), and the implemented modelling attacks are:
 - Supervised Learning:
   * Basic approach
   * Federated learning approach
   * Majority vote approach
 - Modified Probably Appoximately Correct (PAC) approach
 - IPUF attacks:
   * Naive attack
   * Collaborative attack

## How to cite

M. Ferens, E. Dushku, and S. Kosta, "On the Feasibility of Deep Reinforcement Learning for Modeling Delay-based Physical Unclonable Functions," 2024, *unpublished*.

## Dependencies

The code was tested using the following:
 - Python 3.8
 - pypuf 2.2.0
 - numpy 1.23.1
 - chainerrl 0.8.0
 - tensorflow 2.10.0
 - pandas 1.5.1

## How to use

To run single experiments using DRL DQN algorithm use one of the following (use `--help` for a list of arguments):
```
python3 DQN_attack_XORPUF.py
python3 DQN_replicant_attack_XORPUF.py
python3 DQN_vote_attack_XORPUF.py
python3 DQN_delay_attack_XORPUF.py
python3 DQN_attack_IPUF.py
python3 DQN_collaborative_attack_IPUF.py
```
All previous scripts require in the same directory one the classes defined in `PUF_env.py` and `PUF_delay_env.py`.

To run single experiments using Supervised Learning Logistic Regression (LR) or Multi-Layer Perceptron (MLP), and the splitting attack on IPUF, use on of the following (use `--help` for a list of arguments):
```
python3 LR_attack_XORPUF.py
python3 MLP_attack_XORPUF.py
python3 splitting_LR_attack.py
python3 splitting_MLP_attack.py
```

## References

1. N. Wisiol, C. Gräbnitz, C. Mühl, B. Zengin, T. Soroceanu, N. Pirnay, K. T. Mursi, and A. Baliuka, "pypuf: Cryptanalysis of Physically Unclonable Functions," 2021, version v2. [Online]. Available: https://doi.org/10.5281/zenodo.3901410
2. Y. Fujita, P. Nagarajan, T. Kataoka, and T. Ishikawa, "ChainerRL: A Deep Reinforcement Learning Library," in *Journal of Machine Learning Research*, vol. 22, no. 77, pp. 1-14, 2021.
3. V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level Control through Deep Reinforcement Learning," in *Nature*, vol. 518, pp. 529-533, 2015.
