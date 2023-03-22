import argparse
import matplotlib.pyplot as plt
from ode_functions import *


def run(a, x0, method, T, h):
    t = np.arange(0, T + h, h)
    ode = def_ode(a)
    sol = exact_sol(a, x0)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(len(x) - 1):
        if method == "euler":
            x[i + 1] = euler(t[i], x[i], ode, h)
        elif method == "rk4":
            x[i + 1] = rk4(t[i], x[i], ode, h)
        else:
            raise "method must be euler or rk4"

    fig, ax = plt.subplots()
    ax.scatter(t, x, label=f'{method}')
    ax.plot(t, sol(t), label='exact')
    ax.legend()
    ax.set_xlim(0, T)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=float)
    parser.add_argument("x0", type=float)
    parser.add_argument("method", type=str)
    parser.add_argument("T", type=int)
    parser.add_argument("--h", type=float, default=0.1)
    args = parser.parse_args()
    run(args.a, args.x0, args.method, args.T, args.h)
