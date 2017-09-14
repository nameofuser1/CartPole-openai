# Deep Q-learning
This is an implementation of s Double Deep Q-Network and prioritized experience replay, applied to OpenAI CartPole task. Can be easily modified to be used with any other environment.

# Known issues
For some reason model tends to degrade fast after ~50 epochs(still working on it). Nonetheless until it happens, model shows a good learning. Model is rather dependent on initial memory state which leads to different learning efficiency. However introduction of DDQN made it to converge anyway. 

# Future plans
There is still room for experimenting. For instance with Dueling DQNs which will be introduced in the nearest future.

License
----

MIT
