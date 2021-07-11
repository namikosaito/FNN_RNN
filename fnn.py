# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import math


class layer(metaclass=ABCMeta):
    @abstractmethod
    def forward_propagation(self, x):
        pass

    @abstractmethod
    def backward_propagation(self, du):
        pass

    def update_params(self):
        pass

    def show_params(self):
        pass


class Affin(layer):
    def __init__(self, w_shape, learn_rate):
        self._learn_rate = learn_rate
        normal_scale = 1 / math.sqrt(w_shape[0])
        self._w = np.random.normal(scale=normal_scale, size=w_shape)
        self._b = np.random.normal(scale=normal_scale, size=(w_shape[0], 1))

    def forward_propagation(self, x):
        self._x = x
        ret = np.dot(self._w, x) + self._b
        # ret = (np.dot(self._w, x) + self._b) * self._tau + ret_old * (1 - self._tau)
        return ret

    def backward_propagation(self, du):
        round_L_x = np.dot(self._w.T, du)
        self._round_L_w = np.dot(du, self._x.T)
        self._round_L_b = du
        return round_L_x

    def update_params(self):
        self._w -= self._learn_rate * self._round_L_w
        self._b -= self._learn_rate * self._round_L_b

    def show_params(self):
        print("w: ", self._w)
        print("b: ", self._b)


class Relu(layer):
    def forward_propagation(self, x):
        self._x = x
        return np.maximum(0, x)

    def backward_propagation(self, du):
        return du * np.where(self._x > 0, 1, 0)


class MeanSquaredError(layer):
    def forward_propagation(self, x, t):
        self._diff = x - t
        t_np = np.array(t)
        if t_np.shape:
            self._n = t_np.shape[0]
        else:
            self._n = 1
        ret = np.sum(np.power(self._diff, 2)) / self._n
        return ret

    def backward_propagation(self):
        return 2 / self._n * self._diff


class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._epoch = 1

    def init_params(self, epoch):
        self._epoch = epoch

    def add(self, layer):
        self._layers.append(layer)

    def set_error_layer(self, layer):
        self._error_layer = layer

    def fit(self, x, y):
        layer_num = len(self._layers)
        for _ in range(self._epoch):
            for one_data, teach_data in zip(x, y):
                target_x = one_data
                for i in range(layer_num):
                    target_x = self._layers[i].forward_propagation(target_x)
                self._error_layer.forward_propagation(target_x, teach_data)
                du = self._error_layer.backward_propagation()
                for i in reversed(range(layer_num)):
                    du = self._layers[i].backward_propagation(du)
                    self._layers[i].update_params()

    def predict(self, x):

        pred_y = np.empty(shape=x.shape[0])
        for k, one_data in enumerate(x):
            target_x = one_data
            for i in range(len(self._layers)):
                target_x = self._layers[i].forward_propagation(target_x)
            pred_y[k] = target_x
        return pred_y

    def get_error(self):
        return self._loss

    def show_params(self):
        for layer in self._layers:
            layer.show_params()


if __name__ == "__main__":

    network = NeuralNetwork()
    network.init_params(epoch=1000)
    network.add(Affin(w_shape=[10, 1], learn_rate=0.001))
    network.add(Relu())
    network.add(Affin(w_shape=[10, 10], learn_rate=0.001))
    network.add(Relu())
    network.add(Affin(w_shape=[1, 10], learn_rate=0.001))
    network.set_error_layer(MeanSquaredError())

    # y=sin(x)  (-pi < x <pi)を母集団とするデータを用いる。trainとtestに分ける
    x_all = np.arange(-314, 314) * 0.01
    data_num = len(x_all)

    np.random.shuffle(x_all)
    train_num = int(data_num * 0.5)

    x_train = x_all[:train_num]
    x_test = x_all[train_num:]

    y_train = np.sin(x_train)
    y_test = np.sin(x_test)

    network.fit(x_train, y_train)

    pred_y_train = network.predict(x_train)
    pred_y = network.predict(x_test)

    # network.show_params()
    cal_error = MeanSquaredError()
    print("train MSE: ", cal_error.forward_propagation(pred_y_train, y_train))
    pred_error = cal_error.forward_propagation(pred_y, y_test)
    print("test MSE: ", pred_error)
    plt.scatter(x_test, pred_y, c="red", label="pred")
    plt.scatter(x_test, y_test, c="blue", label="y=sin(x)")
    plt.legend()
    plt.show()
