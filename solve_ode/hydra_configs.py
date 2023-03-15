import matplotlib.pyplot as plt
from ode_functions import *
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig) -> None:

    t = np.arange(0, cfg.T + cfg.h, cfg.h)
    ode = def_ode(cfg.a)
    sol = exact_sol(cfg.a, cfg.x0)
    x = np.zeros_like(t)
    x[0] = cfg.x0
    for i in range(len(x) - 1):
        x[i + 1] = hydra.utils.call(cfg.method, t[i], x[i], ode)

    fig, ax = plt.subplots()
    ax.scatter(t, x)
    ax.plot(t, sol(t))
    ax.set_xlim(0, cfg.T)
    plt.show()


if __name__ == "__main__":
    run()
