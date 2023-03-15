import matplotlib.pyplot as plt
from ode_functions import *


def run():
    a = 1
    h = 0.1
    T = 2
    x0 = 1

    t = np.arange(0, T + h, h)
    ode = def_ode(a)
    sol = exact_sol(a, x0)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(len(x) - 1):
        x[i + 1] = euler(t[i], x[i], ode, h)

    fig, ax = plt.subplots()
    ax.scatter(t, x)
    ax.plot(t, sol(t))
    ax.set_xlim(0, T)
    plt.show()


if __name__ == "__main__":
    run()
