from agent import CartpoleAgent
from memory import PrioritizedMemory
from net import KerasQNet
import numpy as np
import gym

import matplotlib.pyplot as plt


EPISODES = 10000

STATE_SIZE = 4
STATE_INDICES = (0, 1, 2, 3)
ACTION_SPACE_SIZE = 2
MEMORY_SIZE = 512

REWARD_TYPE = 'side_rewards'

ACTION_LEFT = 0
ACTION_RIGHT = 1

LR = 1e-3
C = 20.


# SAMPLE FORM
# (state, next_state, action, reward, done, steps)

def reshape_state(s):
    return np.reshape(np.take(s, STATE_INDICES), (1, STATE_SIZE))


def env_step(env, a):
    s, r, d, _ = env.step(a)
    s = reshape_state(s)

    return s, r, d, _


def exponential_decay_coeff(theta, c):
    b = c + 1.
    k = np.log(b)/c

    return (-np.exp(k*theta) + b) / c


def side_rewards(sample, c):
    action = sample[2]
    theta = sample[0][0][2]

    if ((theta < 0) and (action == ACTION_LEFT))\
            or ((theta > 0) and (action == ACTION_RIGHT)) or (theta == 0):

        sample[3] *= c

    return sample


def exponential_decay_rewards(sample, c):
    s0 = sample[0][0]
    x = s0[0]
    theta = s0[2]
    # a = sample[2]
    # done = sample[4]
    # steps = sample[5]

    # From 0 to 15
    theta_deg = min(abs(theta*180./np.pi), 15.)
    theta_k = exponential_decay_coeff(theta_deg, 11.)
    xk = exponential_decay_coeff(abs(x), 2.4)

    sample[3] *= c*xk*theta_k
    return sample


REWARDS = {'side_rewards': side_rewards,
           'exp_rewards': exponential_decay_rewards}


def process_sample(sample, c=10., rew_type='side_rewards'):
    return REWARDS[rew_type](sample, c)


def fill_memory(env, agent):
    s = reshape_state(env.reset())
    steps = 0

    while memory.capacity != memory.size():
        a = env.action_space.sample()
        s1, r, d, _ = env_step(env, a)
        steps += 1

        sample = process_sample([s, s1, a, r, d, steps], c=C,
                                rew_type=REWARD_TYPE)
        agent.memorize(sample)

        s = s1

        if d:
            steps = 0
            env.reset()

    return memory


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    hidden_sizes = [8]
    net = KerasQNet()
    net = net.build_net(STATE_SIZE, hidden_sizes, ACTION_SPACE_SIZE,
                        lr=LR)
    memory = PrioritizedMemory(capacity=MEMORY_SIZE)

    agent = CartpoleAgent(net, env.action_space, ACTION_SPACE_SIZE,
                          STATE_SIZE, memory, min_explore_rate=0.1)

    fill_memory(env, agent)

    rewards = []
    losses = []

    try:
        env.reset()
        gym.wrappers.Monitor(env, 'monitor/recoring-2', force=True)

        for i in range(EPISODES):
            s = reshape_state(env.reset())
            done = False

            steps = 0
            episode_reward = 0

            while not done:
                env.render()
                a = agent.act(s)
                s1, r, done, _ = env_step(env, a)
                steps += 1

                sample = process_sample([s, s1, a, r, done, steps],
                                        c=C, rew_type=REWARD_TYPE)
                agent.memorize(sample)
                losses.append(agent.replay())

                s = s1
                episode_reward += r

            # Save for some statistics
            rewards.append(episode_reward)
            print("Episode %d reward: %f" % (i, episode_reward))

            if i % 100 == 0 and i != 0:
                mean_r = np.mean(rewards[-100])

                if mean_r >= 475:
                    print("Training finished on %d episode" % i)
                    break

    finally:
        net.save("models/keras-qmodel2.h5")

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(rewards)
        ax2 = fig.add_subplot(212)
        ax2.plot(losses)
        plt.tight_layout()

        fig.savefig('models/info1.png', dpi=fig.dpi)
