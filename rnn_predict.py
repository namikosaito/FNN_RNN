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


class Sigmoid(layer):
    def forward_propagation(self, x):
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward_propagation(self, du):
        return du * self._y * (1 - self._y)


class Tanh(layer):
    def forward_propagation(self, x):
        self._x = x
        return np.tanh(x)

    def backward_propagation(self, du):
        return du * 4 / ((np.exp(self._x) + np.exp(-self._x)) ** 2)


class RnnLayer(layer):
    def __init__(self, w_shape, learn_rate, active_func):
        self._learn_rate = learn_rate
        # Xivierの初期値を用いる
        w_in_normal_scale = 1 / math.sqrt(w_shape[0])
        w_recorded_normal_scale = 1 / math.sqrt(w_shape[0])
        self._w_in = np.random.normal(scale=w_in_normal_scale, size=w_shape)
        self._w_recorded = np.random.normal(
            scale=w_recorded_normal_scale, size=(w_shape[0], w_shape[0])
        )
        self._b = np.random.normal(scale=w_in_normal_scale, size=(w_shape[0], 1))

        self._active_func = active_func
        self._ret = None

    def forward_propagation(self, x):
        self._x = x
        self._old_ret = self._ret  # oldの値はBPTTでも使用するため、forwardの計算直前で更新
        if self._old_ret is None:  # 初回のみ普通のDense層
            # if x.shape:
            #     x_shape = x.shape[0]
            # else:
            #     x_shape = 1
            self._old_ret = np.zeros(shape=(self._w_in.shape[0], 1))
        u = np.dot(self._w_in, x) + np.dot(self._w_recorded, self._old_ret) + self._b
        self._ret = self._active_func.forward_propagation(u)
        # ret = (np.dot(self._w, x) + self._b) * self._tau_reverse + ret_old * (1 - self._tau_reverse) 時定数を入れたCTRNNの場合
        return self._ret

    def backward_propagation(self, du):
        du_new = self._active_func.backward_propagation(du)
        round_L_x = np.dot(self._w_in.T, du_new)
        self._round_L_w_in = np.dot(du_new, self._x.T)
        self._round_L_w_recorded = np.dot(du_new, self._old_ret.T)
        self._round_L_b = du_new
        return round_L_x

    def update_params(self):
        self._w_in -= self._learn_rate * self._round_L_w_in
        self._w_recorded -= self._learn_rate * self._round_L_w_recorded
        self._b -= self._learn_rate * self._round_L_b

    def show_params(self):
        print("w: ", self._w)
        print("b: ", self._b)


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
        layer_num_list = range(len(self._layers))
        for _ in range(self._epoch):
            for one_data, teach_data in zip(x, y):
                u = one_data
                for i in layer_num_list:
                    u = self._layers[i].forward_propagation(u)
                self._error_layer.forward_propagation(u, teach_data)
                du = self._error_layer.backward_propagation()
                for i in reversed(layer_num_list):
                    du = self._layers[i].backward_propagation(du)
                    self._layers[i].update_params()

    def predict(self, x):
        pred_y = []
        for k, one_data in enumerate(x):
            target_x = one_data
            for i in range(len(self._layers)):
                target_x = self._layers[i].forward_propagation(target_x)
            pred_y.append(target_x)
        return np.array(pred_y)

    def get_error(self):
        return self._loss

    def show_params(self):
        for layer in self._layers:
            layer.show_params()


def target_func(t):
    return np.array([[2 * math.sin(t)], [2 * math.sin(2 * t)]])


def make_data(start, data_num, step):
    x = []
    y = []
    t_list = [t * step for t in range(int(start / step), int(start / step + data_num))]
    for t in t_list:
        x.append(target_func(t))
        y.append(target_func(t + step))
    return np.array(x), np.array(y)


if __name__ == "__main__":

    network = NeuralNetwork()
    network.init_params(epoch=100)
    network.add(RnnLayer(w_shape=[10, 2], learn_rate=0.01, active_func=Tanh()))
    network.add(RnnLayer(w_shape=[10, 10], learn_rate=0.01, active_func=Tanh()))
    network.add(Affin(w_shape=[2, 10], learn_rate=0.01))
    network.set_error_layer(MeanSquaredError())

    # network.add(Affin(w_shape=[10, 2], learn_rate=0.01))
    # network.add(Tanh())
    # network.add(Affin(w_shape=[10, 10], learn_rate=0.01))
    # network.add(Tanh())
    # network.add(Affin(w_shape=[2, 10], learn_rate=0.01))
    # network.set_error_layer(MeanSquaredError())

    t_step = 0.01
    x_train, y_train = make_data(0, 628, t_step)
    test_num = 600
    x_test, y_test = make_data(314, test_num, t_step)

    network.fit(x_train, y_train)
    pred_y_train = network.predict(x_train)
    pred_y = network.predict(x_test)
    # 実験2の場合。 予測結果を用いてさらに予測していく
    # pred_y = []
    # x = [x_test[0]]
    # for _ in range(test_num):
    #     x = network.predict(x)
    #     pred_y.append(x[0])
    # pred_y = np.array(pred_y)

    cal_error = MeanSquaredError()
    print("train MSE: ", cal_error.forward_propagation(pred_y_train, y_train))
    pred_error = cal_error.forward_propagation(pred_y, y_test)
    print("test MSE: ", pred_error)
    pred_x_t = [one_data[0] for one_data in pred_y]
    pred_y_t = [one_data[1] for one_data in pred_y]
    test_x_t = [one_data[0] for one_data in y_test]
    test_y_t = [one_data[1] for one_data in y_test]
    fig = plt.figure()
    plt.scatter(pred_x_t, pred_y_t, c="red", label="pred")
    plt.scatter(test_x_t, test_y_t, c="blue", label="test")
    plt.legend()
    fig.savefig("temp.png")
    plt.show()
