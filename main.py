import numpy as np
import random
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1 - x * x


def step(x, target, lr):
    h = [tanh(x[0] * w[0, 0] + x[1] * w[1, 0]), tanh(x[0] * w[0, 1] + x[1] * w[1, 1])]
    out = tanh(h[0] * o[0] + h[1] * o[1])

    error_d = (target - out) * tanh_d(out)

    h_d = [error_d * o[0] * tanh_d(h[0]), error_d * o[1] * tanh_d(h[1])]

    o[0] += lr * h[0] * error_d
    o[1] += lr * h[1] * error_d

    w[0, 0] += lr * x[0] * h_d[0]
    w[0, 1] += lr * x[0] * h_d[1]
    w[1, 0] += lr * x[1] * h_d[0]
    w[1, 1] += lr * x[1] * h_d[1]


def pred(x):
    h = [tanh(x[0] * w[0, 0] + x[1] * w[1, 0]), tanh(x[0] * w[0, 1] + x[1] * w[1, 1])]
    out = tanh(h[0] * o[0] + h[1] * o[1])
    return out


def train():
    mse_ = []
    epoch_ = []
    for epoch in range(epochs):
        for pair in input_data:
            step(pair, output_data[input_data.index(pair)], learning_rate)

        if epoch % 100 == 0:
            mse = 0
            for pair in input_data:
                result = pred(pair)
                mse += ((output_data[input_data.index(pair)] - result) ** 2) / 4
            mse_.append(mse)
            epoch_.append(epoch)

    for pair in input_data:
        result = pred(pair)
        print(f'Input: {pair}, excepted: {output_data[input_data.index(pair)]}, got: {round(result, 3)}')

    plt.plot(epoch_, mse_)
    plt.xlabel('Epochs')
    plt.ylabel('Mean square error')
    plt.grid(visible=True)
    plt.show()


def classification(x, y):
    if pred([x, y]) >= 0.5:
        return 1


def plots():
    plt.axis('scaled')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(visible=True)

    plt.plot([0], [0], color='gray', markersize=15, marker='o')
    plt.plot([0], [1], color='red', markersize=15, marker='o')
    plt.plot([1], [0], color='red', markersize=15, marker='o')
    plt.plot([1], [1], color='gray', markersize=15, marker='o')

    x_range = np.arange(-0.1, 1.1, 0.05)
    y_range = np.arange(-0.1, 1.1, 0.05)

    x_, y_ = np.meshgrid(x_range, y_range, indexing='xy')
    z = np.array([[classification(x, y) for x in x_range] for y in y_range])
    plt.contourf(x_, y_, z, colors=['red'], alpha=0.5)
    plt.show()


input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [0, 1, 1, 0]

var = 0.5
learning_rate = 0.05
epochs = 5001

w = np.array([[random.uniform(-var, var), random.uniform(-var, var)], [random.uniform(-var, var), random.uniform(-var, var)]])
o = np.array([random.uniform(-var, var), random.uniform(-var, var)])

train()
plots()
