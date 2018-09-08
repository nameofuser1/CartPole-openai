import numpy as np
import logging


class CartpoleAgent(object):

    def __init__(self, brain, action_space, action_space_size,
                 state_size, memory, start_explore_rate=1.,
                 explore_decay=0.99, min_explore_rate=0.05,
                 gamma=0.95, update_target_freq=10.):
        """
        brain       ---     instance of a network class(net.py)

        """

        # Q-networks
        self._net = brain
        self._target_net = brain.copy()

        # For sampling and learning
        self._action_space = action_space
        self._action_space_size = action_space_size
        self._state_size = state_size

        # Memory for experience replay
        self._memory = memory

        # Next-state Q-value factor
        self._gamma = gamma

        # Explorations parameters
        self._explore_rate = start_explore_rate
        self._explore_decay = explore_decay
        self._min_explore_rate = min_explore_rate

        # Params for updating target network
        self._update_target_freq = update_target_freq
        self._target_steps = 0

        fhandler = logging.FileHandler("logs/agent.log", 'w')
        fhandler.setLevel(logging.DEBUG)

        self._logger = logging.getLogger(__name__ + "_all")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        self._logger.addHandler(fhandler)

    def _update_target_net(self):
        self._target_net.weights = self._net.weights

    def act(self, s):
        if np.random.rand() <= self._explore_rate:
            action = self._action_space.sample()
        else:
            action = np.argmax(self._net.predict(s))

        return action

    def memorize(self, sample):
        """
        sample = [s, s1, a, r]

        s   --- initial state
        s1  --- next state
        a   --- action to get [s -> s1]
        r   --- reward for that action
        """
        s = sample[0]
        s1 = sample[1]
        a = sample[2]
        r = sample[3]

        # Current q-values for state S
        q_target = self._net.predict(s)

        # Double DQN
        # https://papers.nips.cc/paper/3964-double-q-learning.pdf Double
        #   q-learning
        # https://arxiv.org/pdf/1509.06461.pdf DDQN
        primary_net_q = self._net.predict(s1)
        target_net_q = self._target_net.predict(s1)

        # Choose action and its' q-values accroding to out primary network
        best_a = np.argmax(primary_net_q)
        action_q = r + self._gamma*target_net_q[best_a]

        dq = np.abs(q_target[a] - action_q)
        q_target[a] = action_q

        # Memorize with dq as priority weight
        item = (dq, (q_target, sample))
        self._memory.memorize(item)

    def replay(self, batch_size=32):
        """
        Performs batch training. Batch is retrieved from memory
        """
        batch = self._memory.remember(batch_size)
        self._logger.debug("Training on batch:\r\n" + str(batch) + "\r\n")

        x, y = self.__get_targets(batch)
        hist = self._net.train(x, y, batch_size=batch_size)

        self._target_steps += 1

        if self._target_steps == self._update_target_freq:
            self._update_target_net()
            self._target_steps = 0

        self._explore_rate = max(self._min_explore_rate,
                                 self._explore_rate*self._explore_decay)

        return np.mean(hist.history['loss'])

    def __get_targets(self, batch):
        """
        batch = [(weight0, (q0, sample0)), (weight1, (q1, sample1)), ...]

        weight  --- indicates how useful this training sample is
        q       --- target QValues
        sample  --- training sample containing [s, s1, a, r, d, steps]
        """
        x = np.zeros((len(batch), self._state_size),
                     dtype=np.float32)
        y = np.zeros((len(batch), self._action_space_size),
                     dtype=np.float32)

        k = 0
        put_xrange = range(self._state_size)
        put_yrange = range(self._action_space_size)

        for weight, item in batch:
            q_target, sample = item
            s, s1, a, r, d, steps = sample

            np.put(x[k], put_xrange, s)
            np.put(y[k], put_yrange, q_target)
            k += 1

        return x, y
