from abc import ABCMeta, abstractmethod, abstractproperty
from utils import abstractstatic
import keras.backend as K
from keras.models import Sequential, load_model, model_from_json
from keras import losses
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

    @abstractstatic
    def load(fname):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractproperty
    def weights(self):
        pass


def simple_qloss(y_true, y_pred):
    err = y_true - y_pred
    return K.mean(K.square(err))

# Fix for loading models with custom loss function
# In future have to replace with built-in MSE loss
losses.simple_qloss = simple_qloss


class KerasNet(Net):

    def __init__(self, model):
        self._model = model

    def save(self, fname):
        self._model.save(fname)

    @staticmethod
    def load(fname):
        net = KerasQNet(model=load_model(fname))
        return net

    def copy(self):
        model = model_from_json(self._model.to_json())
        return KerasQNet(model=model)

    def train(self, x, y, batch_size=32, epochs=1, verbose=0):
        return self._model.fit(x, y, batch_size=batch_size, verbose=verbose)

    def predict(self, x):
        return self._model.predict(x).flatten()

    @property
    def weights(self):
        return self._model.get_weights()

    @weights.setter
    def weights(self, w):
        self._model.set_weights(w)


class KerasQNet(KerasNet):

    def __init__(self, model=None):
        super(KerasQNet, self).__init__(model=model)

    @staticmethod
    def build_net(input_size, hidden_sizes, output_size,
                  lr=0.01, loss=simple_qloss):
        model = Sequential()

        if hidden_sizes is None:
            model.add(Dense(output_size, input_dim=input_size,
                            activation='linear'))

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

        return KerasQNet(model=model)
