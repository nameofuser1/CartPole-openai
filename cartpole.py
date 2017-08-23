import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import sleep
from Threaded import Threaded


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

    def __init__(self, model_name='model1'):
        self.env = gym.make('CartPole-v1')

        hidden_sizes = [8, 2]
        input_, action_prob, gradients, gradient_holders,\
            updateBatch, loss = self.create_network(4, 1e-2, *hidden_sizes)

        self.input_ = input_
        self.action_prob = action_prob
        self.gradients = gradients
        self.gradient_holders = gradient_holders
        self.updateBatch = updateBatch
        self.loss = loss

        self.model_name = model_name

        print("Network created")

    def discounted_rewards(self, r, gamma=0.99):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

        return discounted_r

    def modified_rewards0(self, history, mult=100):
        """
        Poor results
        """
        state_id = ThreadedCartpole.STATE_ID
        action_id = ThreadedCartpole.ACTION_ID
        reward_id = ThreadedCartpole.REWARD_ID

        states = history[:, state_id]
        actions = history[:, action_id]
        rewards = history[:, reward_id]

        discounted_r = np.zeros_like(rewards)

        idx = 0
        for s, a, r in zip(states, actions, rewards):
            nrew = 0
            x = s[0]
            theta = s[2]

            if ((theta < 0) and (a == 1)) or ((theta > 0) and (a == 0)):
                nrew = -np.abs(x)*r*mult
            else:
                nrew = min(500, 1./abs(x))*r

            discounted_r[idx] = nrew
            idx += 1

        return discounted_r

    def compute_loss(self, tf_output):
        """
        Total number of actions(n) is the same as the first entry of output
            shape since it is equal to number of inputs. Each output contains 2
            probabilities for each action. Flattening the output we get 2*n
            entries every two corresponding to one action. Therefore we can
            choose our probability by summing even number in range of 2*n with
            our actions which is 0 or 1.
        """
        self.reward_holder = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=(None,), dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(tf_output)[0])\
            * tf.shape(tf_output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(tf_output, [-1]),
                                             self.indexes)

        # Minus for optimality
        loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                               self.reward_holder)

        return loss

    def create_network(self, in_size=4, lr=1e-2, *args):
        """
        Network:
            2 inputs
            First hidden layer 32 neurons
            Second hidden layer 16 neurons
            2 outputs (left, right)
        """
        # Input layer
        input0 = tf.placeholder(shape=[None, in_size], dtype=tf.float32)

        activation = tf.nn.relu
        hidden = input0
        print(args)
        print(len(args))
        for idx, size in enumerate(args):
            print("IDX: " + str(idx))
            if idx == len(args) - 1:
                print("Changed activation")
                activation = tf.nn.softmax

            hidden = slim.fully_connected(hidden, size,
                                          activation_fn=activation,
                                          biases_initializer=None)

        action_prob = hidden
        print(input0)
        print(action_prob)

        # LOSS
        loss = self.compute_loss(action_prob)

        # Computing gradients
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars)

        gradient_holders = []
        for idx, tvar in enumerate(tvars):
            gradient_holders.append(tf.placeholder(dtype=tf.float32,
                                                   name=str(idx)+"_holder"))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        updateBatch = optimizer.apply_gradients(zip(gradient_holders, tvars))

        return (input0, action_prob, gradients, gradient_holders,
                updateBatch, loss)

    def __choose_action(self, probs):
        """
        Probalistically choose action based on computed
            action probabilities
        """
        action_prob = np.random.choice(probs, p=probs)
        chosen_action = np.argmax(probs == action_prob)

        return chosen_action

    def __clear_gradient_buffers(self, buf):
        for idx, grad in enumerate(buf):
            buf[idx] = grad*0.0

        return buf

    def __prepare_gradient_buffers(self, sess):
        """
        Since we don't know dimensions beforehand, we compute trainable
            variables and get list of numpy arrays of needed
            dimensions(each variable gives one gradient d(loss)/d(var))
        """
        gradient_buffer = sess.run(tf.trainable_variables())
        gradient_buffer = self.__clear_gradient_buffers(gradient_buffer)

        return gradient_buffer

    def train(self, episodes=5000, max_steps=1000, episodes_per_log=100):
        """
        """
        self.env.reset()
        self.renderer()

        with tf.Session() as sess:
            try:
                sess.run(tf.initialize_all_variables())

                gradients_buffer = self.__prepare_gradient_buffers(sess)
                rewards = []

                for i in range(episodes):
                    s = self.env.reset()
                    episode_reward = 0
                    np_history = []

                    for j in range(max_steps):
                        feed_dict = {
                            self.input_: [s]
                        }

                        action_probs = sess.run(self.action_prob,
                                                feed_dict=feed_dict)[0]
                        action = self.__choose_action(action_probs)

                        s1, r, d, _ = self.env.step(action)
                        np_history.append([s, s1, action, r])

                        s = s1
                        episode_reward += r

                        if d:
                            state_id = ThreadedCartpole.STATE_ID
                            action_id = ThreadedCartpole.ACTION_ID
                            reward_id = ThreadedCartpole.REWARD_ID

                            rewards.append(episode_reward)

                            np_history = np.array(np_history)
                            np_history[:, reward_id] =\
                                self.modified_rewards0(np_history)

                            feed_dict = {
                                # Do we really need vstack? Tested with numpy
                                # seems like slice is already stacked
                                self.input_: np.vstack(np_history[:, state_id]),
                                self.action_holder: np_history[:, action_id],
                                self.reward_holder: np_history[:, reward_id]
                            }

                            # Compute gradients
                            gradients = sess.run(self.gradients,
                                                 feed_dict=feed_dict)

                            for idx, grad in enumerate(gradients):
                                gradients_buffer[idx] += grad

                            if (i % EPISODES_PER_UPDATE == 0) and (i != 0):
                                feed_dict = dict(zip(self.gradient_holders,
                                                     gradients_buffer))

                                sess.run(self.updateBatch, feed_dict=feed_dict)
                                gradients_buffer = self.\
                                    __clear_gradient_buffers(gradients_buffer)

                            break

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
