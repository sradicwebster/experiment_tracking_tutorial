import argparse
import wandb
import yaml


def run(config):
    with open(f"configs/sweeps/{config}.yaml", 'r') as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
    wandb.agent(sweep_id, project=sweep_config["project"], count=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    run(args.config)
