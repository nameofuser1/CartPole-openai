from agent import CartpoleAgent
from memory import SimpleMemory
from net import KerasQNet
import numpy as np
import gym

from Threaded import Threaded
from time import sleep


EPISODES = 10000

STATE_SIZE = 2
ACTION_SPACE_SIZE = 2
MEMORY_SIZE = 512

ACTION_LEFT = 0
ACTION_RIGHT = 1

C = 100.


def process_sample(sample, c=10.):
    theta = sample[0][2]
    a = sample[2]

    if (theta < 0 and a == ACTION_LEFT) or (theta > 0 and a == ACTION_RIGHT):
        sample[3] *= c

    sample[0] = sample[0][2:]
    sample[1] = sample[1][2:]

    return sample


def fill_memory(env, memory):
    s = env.reset()

    while memory.capacity != memory.size():
        a = env.action_space.sample()
        s1, r, d, _ = env.step(a)

        sample = process_sample([s, s1, a, r], c=C)
        memory.memorize(sample)

        s = s1

        if d:
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

            while not done:
                a = agent.act(s)
                s1, r, done, _ = env.step(a)

                sample = process_sample([s, s1, a, r], c=C)
                agent.memorize(sample)
                agent.replay()

                s = s1
                episode_reward += r

            # Save for some statistics
            rewards.append(episode_reward)

            if i % 100 == 0 and i != 0:
                print("Passed %d episodes" % i)
                print("Mean reward for last 100: %f" %
                      (np.mean(rewards[-100:])))

    finally:
        net.save("models/keras-qmodel0.h5")
        Threaded.stop_thread("RenderThread")
