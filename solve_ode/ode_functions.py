import numpy as np


def def_ode(a):
    return lambda t, x: a * x


def exact_sol(a, x0):
    return lambda t: x0 * np.exp(a * t)


def exact(t, x, ode, a, x0):
    return x0 * np.exp(a * t)


def euler(t, x, ode, h):
    return x + h * ode(t, x)


def rk4(t, x, ode, h):
    k1 = ode(t, x)
    k2 = ode(t + h / 2, x + h * k1 / 2)
    k3 = ode(t + h / 2, x + h * k2 / 2)
    k4 = ode(t + h / 2, x + h * k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6