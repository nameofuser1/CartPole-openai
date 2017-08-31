import numpy as np


class CartpoleAgent(object):

    def __init__(self, brain, action_space, action_space_size,
                 state_size, memory, start_explore_rate=1.,
                 explore_decay=0.95, min_explore_rate=0.05,
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

    def act(self, s):
        if np.random.rand() <= self._explore_rate:
            action = self._action_space.sample()
        else:
            action = np.argmax(self._net.predict(s))

        return action

    def memorize(self, s):
        self._memory.memorize(s)

    def replay(self, batch_size=32):
        batch = self._memory.remember(batch_size)
        x, y = self.__get_targets(batch)
        self._net.train(x, y, batch_size=batch_size)

    def __get_targets(self, batch):
        x = np.zeros((len(batch), self._state_size),
                     dtype=np.float32)
        y = np.zeros((len(batch), self._action_space_size),
                     dtype=np.float32)

        k = 0
        put_xrange = range(self._state_size)
        put_yrange = range(self._action_space_size)
        for s, s1, a, r in batch:
            s = np.reshape(s, (1, self._state_size))
            s1 = np.reshape(s, (1, self._state_size))
            q_target = self._net.predict(s)
            action_q = r + self._gamma*max(self._net.predict(s1))

            q_target[a] = action_q
            np.put(x[k], put_xrange, s)
            np.put(y[k], put_yrange, q_target)

        return x, y
