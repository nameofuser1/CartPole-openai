from agent import CartpoleAgent
from memory import SimpleMemory
from net import KerasQNet
import numpy as np
import gym

from Threaded import Threaded
from time import sleep


EPISODES = 5000

STATE_SIZE = 4
ACTION_SPACE_SIZE = 2
MEMORY_SIZE = 2000


def fill_memory(env, memory):
    s = env.reset()

    while memory.capacity != memory.size():
        a = env.action_space.sample()
        s1, r, d, _ = env.step(a)

        sample = (s, s1, a, r)
        memory.memorize(sample)

        s = s1

        if d:
            env.reset()

    return memory


def reshape_state(s):
    s1 = np.reshape(np.array(s), (4,))
    return s1


@Threaded.infinite("RenderThread")
def renderer(env):
    env.render()
    sleep(0.02)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    hidden_sizes = [8, 4]
    net = KerasQNet()
    net = net.build_net(STATE_SIZE, hidden_sizes, ACTION_SPACE_SIZE)
    memory = SimpleMemory(MEMORY_SIZE)
    memory = fill_memory(env, memory)

    agent = CartpoleAgent(net, env.action_space, ACTION_SPACE_SIZE,
                          STATE_SIZE, memory)

    try:
        env.reset()
        renderer(env)

        for i in range(EPISODES):
            s = reshape_state(env.reset())
            done = False
            episode_reward = 0

            while not done:
                a = agent.act(s)
                s1, r, done, _ = env.step(a)
                s1 = reshape_state(s1)

                sample = (s, s1, a, r)
                agent.memorize(sample)
                agent.replay()

                episode_reward += 1

            print("Episode %d reward: %f" % (i, episode_reward))

    finally:
        net.save("keras-qmodel0.h5")
        Threaded.stop_thread("RenderThread")
