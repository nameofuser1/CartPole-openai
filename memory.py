from abc import ABCMeta, abstractmethod
from collections import deque
from random import sample
import numpy as np


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
        pass


class SimpleMemory(Memory):

    def __init__(self, capacity=2000):
        self._memory = deque(maxlen=capacity)
        self._capacity = capacity

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

    def size(self):
        return len(self._memory)

    @property
    def capacity(self):
        return self._capacity
