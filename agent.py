import numpy as np
import logging


class CartpoleAgent(object):

    def __init__(self, brain, action_space, action_space_size,
                 state_size, memory, start_explore_rate=1.,
                 explore_decay=0.99, min_explore_rate=0.05,
                 gamma=0.95):

        self._net = brain
        self._action_space = action_space
        self._action_space_size = action_space_size
        self._state_size = state_size
        self._memory = memory
        self._explore_rate = start_explore_rate
        self._explore_decay = explore_decay
        self._min_explore_rate = min_explore_rate
        self._gamma = gamma

        fhandler = logging.FileHandler("logs/agent.log", 'w')
        fhandler.setLevel(logging.DEBUG)

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(fhandler)

    def act(self, s):
        if np.random.rand() <= self._explore_rate:
            action = self._action_space.sample()
        else:
            action = np.argmax(self._net.predict(s))

        return action

    def memorize(self, sample):
        s = sample[0]
        s1 = sample[1]
        a = sample[2]
        r = sample[3]

        q_target = self._net.predict(s)
        action_q = r + self._gamma*max(self._net.predict(s1))
        dq = np.abs(q_target[a] - action_q)

        q_target[a] = action_q

        # Memorize with dq as priority weight
        item = (dq, (q_target, sample))
        self._memory.memorize(item)

    def replay(self, batch_size=32):
        """
        """
        batch = self._memory.remember(batch_size)
        x, y = self.__get_targets(batch)
        hist = self._net.train(x, y, batch_size=batch_size)

        self._explore_rate = max(self._min_explore_rate,
                                 self._explore_rate*self._explore_decay)

        return np.mean(hist.history['loss'])

    def __get_targets(self, batch):
        """
        batch = [(weight0, (q0, sample0)), (weight1, (q1, sample1)), ...]
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

        return x, y
