# Implementation of DDPG - Deep Deterministic Policy Gradient

Modified from the work of Patrick Emami: [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)

Algorithm and hyperparameter details can be found here: ["Continuous control with deep reinforcement learning" - TP Lillicrap, JJ Hunt et al., 2015](http://arxiv.org/abs/1509.02971)

Tested on [CartPole](https://gym.openai.com/envs/CartPole-v0) & [Pendulum](https://gym.openai.com/envs/Pendulum-v0)

### Requirements
[Gym](https://github.com/openai/gym#installation) and [TensorFlow](https://www.tensorflow.org/install/).

### Modifications
- Removed TFLearn dependency
- Added Ornstein Uhlenbeck noise function
- Added reward discounting
- Works with discrete and continuous action spaces