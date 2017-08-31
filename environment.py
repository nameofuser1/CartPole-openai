from agent import CartpoleAgent
from memory import SimpleMemory
from net import KerasQNet
import numpy as np
import gym

from Threaded import Threaded
from time import sleep


EPISODES = 10000

STATE_SIZE = 4
ACTION_SPACE_SIZE = 2
MEMORY_SIZE = 512


def discounted_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def episode_rerewarding(samples, rewards):
    discounted_r = discounted_rewards(rewards)
    for i in range(len(samples)):
        samples[i][3] = discounted_r[i]

    return samples


def fill_memory(env, memory):
    s = env.reset()
    samples = []
    rewards = []
    printed = False

    while memory.capacity != memory.size():
        a = env.action_space.sample()
        s1, r, d, _ = env.step(a)

        samples.append([s, s1, a, r])
        rewards.append(r)

        s = s1

        if d:
            if not printed:
                print(samples)
            samples = episode_rerewarding(samples, rewards)
            if not printed:
                print(samples)
                printed = True

            for sample in samples:
                memory.memorize(sample)

            env.reset()

    return memory


@Threaded.infinite("RenderThread")
def renderer(env):
    env.render()
    sleep(0.02)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    hidden_sizes = [16, 8]
    net = KerasQNet()
    net = net.build_net(STATE_SIZE, hidden_sizes, ACTION_SPACE_SIZE)
    memory = SimpleMemory(MEMORY_SIZE)
    memory = fill_memory(env, memory)

    agent = CartpoleAgent(net, env.action_space, ACTION_SPACE_SIZE,
                          STATE_SIZE, memory)

    try:
        env.reset()
        renderer(env)

        rewards = []

        for i in range(EPISODES):
            s = env.reset()
            done = False

            episode_reward = 0
            samples = []
            episode_rewards = []

            while not done:
                a = agent.act(s)
                s1, r, done, _ = env.step(a)

                agent.replay()

                episode_rewards.append(r)
                samples.append([s, s1, a, r])

                episode_reward += r

            samples = episode_rerewarding(samples, episode_rewards)
            for sample in samples:
                memory.memorize(sample)

            # Save for some statistics
            rewards.append(episode_reward)

            if i % 100 == 0 and i != 0:
                print("Passed %d episodes" % i)
                print("Mean reward for last 100: %f" %
                      (np.mean(rewards[-100:])))

    finally:
        net.save("models/keras-qmodel0.h5")
        Threaded.stop_thread("RenderThread")
