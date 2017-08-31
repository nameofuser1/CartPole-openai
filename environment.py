from agent import CartpoleAgent
from memory import SimpleMemory
from net import KerasQNet
import numpy as np
import gym

from time import sleep
import matplotlib.pyplot as plt


EPISODES = 10000

STATE_SIZE = 2
ACTION_SPACE_SIZE = 2
MEMORY_SIZE = 512

ACTION_LEFT = 0
ACTION_RIGHT = 1

LR = 1e-3
C = 20.


def reshape_state(s):
    return np.reshape(np.asarray(s[2:]), (1, STATE_SIZE))


def env_step(env, a):
    s, r, d, _ = env.step(a)
    s = reshape_state(s)

    return s, r, d, _


def process_sample(sample, c=0., max_c=200.):
    s0 = sample[0][0]
    theta = s0[0]
    a = sample[2]

    # From 0.1 to 15
    # theta_deg = max(abs(theta)*180./np.pi, 0.1)
    # sample[3] *= min(c/theta_deg, max_c)
    # print(sample)

    if (theta < 0 and a == ACTION_LEFT) or (theta > 0 and a == ACTION_RIGHT):
        sample[3] *= c

    return sample


def fill_memory(env, memory):
    s = reshape_state(env.reset())

    while memory.capacity != memory.size():
        a = env.action_space.sample()
        s1, r, d, _ = env_step(env, a)

        sample = process_sample([s, s1, a, r], c=C)
        memory.memorize(sample)

        s = s1

        if d:
            env.reset()

    return memory


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    hidden_sizes = [16, 16]
    net = KerasQNet()
    net = net.build_net(STATE_SIZE, hidden_sizes, ACTION_SPACE_SIZE,
                        lr=LR)
    memory = SimpleMemory(MEMORY_SIZE)
    memory = fill_memory(env, memory)

    agent = CartpoleAgent(net, env.action_space, ACTION_SPACE_SIZE,
                          STATE_SIZE, memory)

    rewards = []
    losses = []

    try:
        env.reset()
        gym.wrappers.Monitor(env, 'monitor/recoring-0', force=True)

        for i in range(EPISODES):
            s = reshape_state(env.reset())
            done = False

            episode_reward = 0

            while not done:
                env.render()
                a = agent.act(s)
                s1, r, done, _ = env_step(env, a)

                sample = process_sample([s, s1, a, r], c=C)
                agent.memorize(sample)
                losses.append(agent.replay())

                s = s1
                episode_reward += r

            # Save for some statistics
            rewards.append(episode_reward)
            print("Episode %d reward: %f" % (i, episode_reward))

            if i % 100 == 0 and i != 0:
                mean_r = np.mean(rewards([-100]))

                if mean_r > 500:
                    print("Training finished on %d episode" % i)
                    break

    finally:
        net.save("models/keras-qmodel1.h5")

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(rewards)
        ax2 = fig.add_subplot(212)
        ax2.plot(losses)
        plt.tight_layout()

        fig.savefig('models/info1.png', dpi=fig.dpi)