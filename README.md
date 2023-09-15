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

## How to cite

M. Ferens, E. Dushku, and S. Kosta, "A Study of Deep Reinforcement Learning applied to Modeling of Delay-based PUFs," 2023, *unpublished*.

## Dependencies

The code was tested using the following:
 - Python 3.8
 - pypuf 2.2.0
 - numpy 1.23.1
 - chainerrl 0.8.0
 - tensorflow 2.10.0
 - pandas 1.5.1

## How to use

To run single experiments use one of the following (use `--help-` for a list of arguments):
```
python3 blablabla.py
...
```
For parametric simulations some examples are available with:
```
python3 main.py
```

## References

1. N. Wisiol, C. Gräbnitz, C. Mühl, B. Zengin, T. Soroceanu, N. Pirnay, K. T. Mursi, and A. Baliuka, "pypuf: Cryptanalysis of Physically Unclonable Functions," 2021, version v2. [Online]. Available: https://doi.org/10.5281/zenodo.3901410
2. Y. Fujita, P. Nagarajan, T. Kataoka, and T. Ishikawa, "ChainerRL: A Deep Reinforcement Learning Library," in *Journal of Machine Learning Research*, vol. 22, no. 77, pp. 1-14, 2021.
3. V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level Control through Deep Reinforcement Learning," in *Nature*, vol. 518, pp. 529-533, 2015.
