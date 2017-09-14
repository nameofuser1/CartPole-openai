from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
from sortedcontainers import SortedList
from random import sample
import numpy as np
from itertools import izip_longest


class Memory(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def memorize(self, state):
        raise NotImplementedError()

    @abstractmethod
    def remember(self):
        raise NotImplementedError()

    @abstractmethod
    def size(self):
        raise NotImplementedError()

    @abstractproperty
    def capacity(self):
        raise NotImplementedError()


class DequeMemory(Memory):

    def __init__(self, capacity=2000):
        self._capacity = capacity
        self._memory = deque(maxlen=capacity)

    def size(self):
        return len(self._memory)

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, v):
        self._capacity = v


def grouper(iterable, n):
    args = [iter(iterable)]*n
    return izip_longest(*args)


class SimpleMemory(DequeMemory):

    def __init__(self, capacity=2000):
        super(SimpleMemory, self).__init__(capacity=capacity)

    def memorize(self, state):
        """
        Deque container will pop first item when maxlen is reached in order
            to append new item.
        """
        self._memory.append(state)

    def remember(self, batch_size=32):
        """
        Returns uniformly distributed samples from memory
        """
        return sample(self._memory, batch_size)


class PrioritizedMemory(DequeMemory):

    def __init__(self, iterable=None, capacity=2000):
        super(PrioritizedMemory, self).__init__(capacity=capacity)

        # Keeps weights in sorted order for sampling
        self._weights = SortedList()

        # Keeps sum of weights
        self._weights_sum = 0

        if iterable is not None:
            for item in iterable:
                self.memorize(item[0], item[1])

    def remember(self, batch_size=32):
        """
        O(n)
        """
        return self.__sample(batch_size)

    def memorize(self, item):
        """
        O(log(n))

        item = (weight, sample)
        """
        weight = item[0]

        if len(self._memory) == self._capacity:
            # O(1) complexity
            discarded = self._memory.popleft()
            self._weights_sum -= discarded[0]
            # O(log(n))
            self._weights.discard(discarded)

        self._memory.append(item)
        self._weights_sum += weight

        # Sorted by weight, O(log(N))
        try:
            self._weights.add(item)
        except ValueError as e:
            print(e)
            print("Trying to memorize item:\r\n\t" + str(item))
            print("Number of items in memory: " +
                    str(self._memory._weights.count(item)))

    def __sample(self, num):
        samples = []
        for i in xrange(num):
            samples.append(self.__sample_one())

        return samples

    def __sample_one(self):
        """
        Sampling one item in O(n)
        """
        rand = np.random.rand() * self._weights_sum
        s = 0
        for i in xrange(self.size()):
            s += self._weights[i][0]

            if s >= rand:
                return self._weights[i]
