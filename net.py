from abc import ABCMeta, abstractmethod
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


class Net(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def build_net(self):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, name):
        pass


def simple_qloss(y_true, y_pred):
    err = y_true - y_pred
    return K.mean(K.square(err))


class KerasQNet(Net):

    def __init__(self):
        self._model = None

    def build_net(self, input_size, hidden_sizes, output_size,
                  lr=0.01, loss=simple_qloss):
        model = Sequential()

        if hidden_sizes is None:
            model.add(Dense(output_size, input_dim=input_size,
                            activation='relu'))

        else:
            activation = 'relu'
            for idx, h in enumerate(hidden_sizes):
                if idx == 0:
                    model.add(Dense(h, input_dim=input_size,
                                    activation=activation))
                else:
                    model.add(Dense(h, activation=activation))

            model.add(Dense(output_size, activation='linear'))

        opt = SGD(lr=lr)
        model.compile(optimizer=opt, loss=loss)

        self._model = model
        return self

    def train(self, x, y, batch_size=32, epochs=1, verbose=0):
        return self._model.fit(x, y, batch_size=batch_size, verbose=verbose)

    def predict(self, x):
        return self._model.predict(x).flatten()

    def save(self, fname):
        self._model.save(fname)
