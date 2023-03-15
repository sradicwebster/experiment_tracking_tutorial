import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import sklearn
from sklearn.model_selection import train_test_split
import wandb


# the @hydra decorator is required and hydra creates a config dictionary from configs/main.yaml
@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig) -> None:

    # this initialises a wandb project with all the config parameters
    wandb.init(project="experiment_tracking_tutorial", config=OmegaConf.to_object(cfg),
               name=cfg.model._target_.split(".")[-1])

    # this creates a instance of the class specified at cfg.dataset._target_ along with specifying
    # the return_X_y argument
    X, y = hydra.utils.instantiate(cfg.dataset, return_X_y=True)

    # sklearn train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_frac, random_state=0)

    # this creates the scaler specified at cfg.scaling._target_ and fits using the training data
    scaler = hydra.utils.instantiate(cfg.scaling).fit(X_train)

    # transform the train and test data
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # instantiates the sklearn model specified at cfg.model._target_ with arguments listed at
    # cfg.model
    model = hydra.utils.instantiate(cfg.model).fit(X_train_s, y_train)

    # call the metric function specified at cfg.metric._target_ with y_true and y_pred arguments to
    # find the training error
    train_error = hydra.utils.call(cfg.metric, y_true=y_train, y_pred=model.predict(X_train_s))

    # find test predictions and error
    test_pred = model.predict(X_test_s)
    test_error = hydra.utils.call(cfg.metric, y_true=y_test, y_pred=test_pred)

    # update the wandb experiment config with error metrics
    wandb.log({"sklearn/train_error": train_error, "sklearn/test_error": test_error})

    # create and a scatter plot of test predictions vs true test labels
    test_table = wandb.Table(data=np.stack((y_test, test_pred), axis=1),
                             columns=["Test", "Predict"])
    wandb.log({"sklearn/test_performance":
                   wandb.plot.scatter(test_table, "Test", "Predict", title="Test Performance")})


if __name__ == "__main__":
    run()
