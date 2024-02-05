import numpy as np 
import matplotlib.pyplot as plt


def f(x: list, gama, B, sig):
    S = x[0]
    E = x[1]
    I = x[2]

    ps = -(B * I * S)
    pe = (B * I * S) - sig * E
    pi = sig * E - gama * I
    pr = gama * I

    return np.array([ps, pe, pi, pr])

def RK4(x, h, gama, sig, B):
    f1 = h * f(x, gama, B, sig)
    f2 = h * f(x + f1 / 2., gama, B, sig)
    f3 = h * f(x + f2 / 2., gama, B, sig)
    f4 = h * f(x + f3, gama, B, sig)

    next_x = x + 1/6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return next_x

def Model_seir(gama, sig, B):
    t0 = 0
    te = 50
    h = .01
    time = np.arange(t0, te, h)

    seir = np.zeros([time.shape[0], 4])
    seir[0, 0] = 0.99
    seir[0, 1] = 0.01
    seir[0, 2] = 0.
    seir[0, 3] = 0.

    for i in range(time.shape[0] - 1):
        seir[i+1] = RK4(seir[i], h, gama, sig, B)

    t = time.shape[0]
    S = [seir[j, 0] for j in range(t)]
    E = [seir[j, 1] for j in range(t)]
    I = [seir[j, 2] for j in range(t)]
    R = [seir[j, 3] for j in range(t)]

    plt.plot(time, S, label='wykres 1')
    plt.plot(time, E, label='wykres 2')
    plt.plot(time, I, label='wykres 3')
    plt.plot(time, R, label='wykres 4')
    plt.xlabel('os X')
    plt.ylabel('os Y')
    plt.legend()
    plt.show()



Model_seir(0.1, 1, 1) 
Model_seir(0.1, 1, 0.5)
