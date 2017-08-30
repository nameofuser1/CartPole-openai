import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import sleep
from Threaded import Threaded
from memory import SimpleMemory


RENDER_POOL_NAME = "CartPoleRender"
FRAMEPERIOD = 0.05

EPISODES_PER_UPDATE = 5


try:
    xrange = xrange
except:
    xrange = range


class ThreadedCartpole(object):

    STATE_ID = 0
    NSTATE_ID = 1
    REWARD_ID = 3
    ACTION_ID = 2

    def __init__(self, model_name='model1', gamma=0.95, explore_rate=1.,
                 explore_decay=0.95, min_explore_rate=0.05):
        """
        gamma               --- discount rate
        explore_rate        --- exploration rate to begin with
        explore_decay       --- explore_rate *= explore_decay for each training
            cycle
        min_explore_rate    --- minimum possible explore_rate
        """
        self.env = gym.make('CartPole-v1')

        self.explore_decay = explore_decay
        self.explore_rate = explore_rate
        self.min_explore_rate = min_explore_rate
        self.gamma = gamma

        hidden_sizes = [8, 2]
        input_, Qout, Qtarget,\
            updateBatch, loss = self.create_network(4, 1e-2, *hidden_sizes)

        self.input_ = input_
        self.Qout = Qout
        self.Qtarget = Qtarget
        self.updateBatch = updateBatch
        self.loss = loss

        # For experience replay
        self._memory = SimpleMemory(memory_size=2000)

        # Model name it will be saved with
        self.model_name = model_name

        print("Network created")

    def discounted_rewards(self, r, gamma=0.99):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

        return discounted_r

    def compute_loss(self, Qout):
        """
        Total number of actions(n) is the same as the first entry of output
            shape since it is equal to number of inputs. Each output contains 2
            probabilities for each action. Flattening the output we get 2*n
            entries every two corresponding to one action. Therefore we can
            choose our probability by summing even number in range of 2*n with
            our actions which is 0 or 1.
        """
        Qtarget = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(Qtarget - Qout))

        return Qtarget, loss

    def create_network(self, in_size=4, lr=1e-2, *args):
        """
        Network:
            2 inputs
            First hidden layer 32 neurons
            Second hidden layer 16 neurons
            2 outputs (left, right)
        """
        # Input layer
        input0 = tf.placeholder(shape=(None, in_size), dtype=tf.float32)

        activation = tf.nn.relu
        hidden = input0
        for idx, size in enumerate(args):
            if idx == len(args) - 1:
                activation = None

            print(hidden)
            hidden = slim.fully_connected(hidden, size,
                                          activation_fn=activation,
                                          biases_initializer=None)
        Qout = hidden
        print(Qout)

        # LOSS
        Qtarget, loss = self.compute_loss(Qout)

        # Optimizing loss function
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        updateBatch = optimizer.apply_gradients(zip(gradients, tvars))

        return (input0, Qout, Qtarget, updateBatch, loss)

    def __reshape_state(self, s):
        return np.array(s).reshape((1, 4))

    def __choose_action(self, Qout):
        """
        Probalistically choose action based on computed
            action probabilities
        """
        if np.random.rand() <= self.explore_rate:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Qout)

        return action

    def __update(self, sess, epochs=32, batch_size=32):
        """
        Performs batch training using experience replay memory
        """
        if self._memory.size() < batch_size:
            return None

        # reward_id = ThreadedCartpole.REWARD_ID

        for i in range(epochs):
            targetQ = []

            memory_batch = self._memory.remember(batch_size=batch_size)
            # memory_batch[:, reward_id] =\
            #    self.discounted_rewards(memory_batch[:, reward_id])

            for s, s1, a, r in memory_batch:
                s1_q = sess.run(self.Qout, feed_dict={self.input_: s1})[0]
                maxq = np.argmax(s1_q)

                # Compute target q-value
                targetQ = r + self.gamma*maxq

                # Compute current q-value
                currentQ = sess.run(self.Qout, feed_dict={self.input_: s})

                # Replace q-value for used action
                currentQ[0][a] = targetQ

                # Train
                sess.run(self.updateBatch, feed_dict={self.input_: s,
                                                      self.Qtarget: currentQ})

        if self.explore_rate > self.min_explore_rate:
            self.explore_rate = max(self.explore_rate*self.explore_decay,
                                    self.min_explore_rate)

    def train(self, episodes=5000, max_steps=1000, episodes_per_log=100):
        """
        """
        self.env.reset()
        self.renderer()

        with tf.Session() as sess:
            try:
                sess.run(tf.initialize_all_variables())
                rewards = []

                for i in range(episodes):
                    s = self.__reshape_state(self.env.reset())
                    episode_reward = 0

                    for j in range(max_steps):
                        if np.random.rand() <= self.explore_rate:
                            action = self.env.action_space.sample()
                        else:
                            Qout = sess.run(self.Qout,
                                            feed_dict={self.input_: s})[0]
                            action = self.__choose_action(Qout)

                        # Next step
                        s1, r, d, _ = self.env.step(action)
                        s1 = self.__reshape_state(s1)

                        # Memorize new state
                        self._memory.memorize((s, s1, action, r))

                        s = s1
                        episode_reward += r

                        if d:
                            rewards.append(episode_reward)
                            break

                    self.__update(sess, epochs=1, batch_size=32)

                    if (i % episodes_per_log == 0) and (i != 0):
                        print("Passed %d episodes" % i)
                        print(np.mean(rewards[-episodes_per_log:]))

            finally:
                saver = tf.train.Saver()
                saver.save(sess, './models/' + self.model_name)
                Threaded.stop_thread(RENDER_POOL_NAME)

    @Threaded.infinite(RENDER_POOL_NAME)
    def renderer(self):
        self.env.render()
        sleep(FRAMEPERIOD)


if __name__ == "__main__":
    tf.reset_default_graph()

    cartpole = ThreadedCartpole()
    cartpole.train()
