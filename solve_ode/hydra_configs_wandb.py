import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import time
from ode_functions import *


@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig) -> None:
    wandb.init(project="experiment_tracking_tutorial", config=OmegaConf.to_object(cfg),
               name=cfg.method._target_.split(".")[-1])
    t = np.arange(cfg.h, cfg.T + cfg.h, cfg.h)
    ode = def_ode(cfg.a)
    x = cfg.x0
    wandb.log({"solve_ode/x": x, "solve_ode/t": 0})
    for i in range(len(t)):
        x = hydra.utils.call(cfg.method, t[i], x, ode)
        wandb.log({"solve_ode/x": x, "solve_ode/t": t[i]})
        time.sleep(0)

    sol = exact_sol(cfg.a, cfg.x0)
    wandb.log({"solve_ode/error": np.abs(x - sol(cfg.T))})


if __name__ == "__main__":
    run()
