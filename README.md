# A Python tutorial on W&B and Hydra to configure, visualise and evaluate experiments

Firstly, create a conda environment, clone the repo and install the required packages:

```zsh
conda create -n experiment_tracking_tutorial
conda activate experiment_tracking_tutorial
conda install pip
git clone git@github.com:sradicwebster/experiment_tracking_tutorial.git
cd experiment_tracking_tutorial
pip install -r requirements.txt
```

[Weights and Biases](https://docs.wandb.ai/) is used to track the experiments. After signing up for a free account, run the following:

```zsh
wandb login
wandb online
```

## Solving ODE

This section covers writing a program to solve the initial value problem via numerical integration for an ordinary differential equation of the form:

$$ \frac{dx}{dt} = f(t, x) = \alpha\ x $$

where $x(t_0)=x_0$.

The exact solution is

$$ x = x_0\ e^{\alpha t} $$ 

which can be used to check the accuracy of the numerical methods.

The Euler method performs the following approximation:

$$ x_{t+1} = x_t + h\ f(t, x_t) $$

where $h$ is the step size.

The more accurate Runge–Kutta method is:

$$ x_{t+1} = x_t + \frac{h}{6} (k_1 + 2 k_2 + 2 k_3 + k_4)$$

where: $k_1 = f(t, x_t)$, $k_2 = f(\frac{t + h}{2}, x_t + h \frac{k_1}{2})$,  $k_3 = f(\frac{t + h}{2}, x_t + h \frac{k_2}{2})$ and $k_4 = f(\frac{t + h}{2}, x_t + h\ k_3)$

These functions have been implemented in `solve_ode/ode_functions.py`.

This tutorial steps through 4 frameworks of increasing complexity building towards combining [Hydra](https://hydra.cc/docs/intro/) configurations and [W&B](https://docs.wandb.ai/) for tracking and visualising solutions.

### Define variables inline

The key variables such as initial values and step size can be defined within the Python script.

```zsh
python solve_ode/define_vars.py
```

The variables are hard coded so to change you have to modify the Python script which is not good practice.

### Command line arguments

To get around this you can use the `argparse` module to allow command line arguments.

```zsh
python solve_ode/command_line_args.py 1 1 euler 2
```

### Hydra configuration

An alternative is to put all the variables into configuration files and use the configuration framework Hydra. This is especially useful when there are a lot of variables to define which fit into natural groups. To use the default values as defined in the configs files, run

```zsh
python solve_ode/hydra_configs.py
```

These values can be overriden in the command line, for example:

```zsh
python solve_ode/hydra_configs.py method=rk4 h=0.01
```

### Hydra configuration and W&B tracking

All the experiments so far have stored the values in a numpy array and produced a matplotlib plot of the results. An alternative is to track the values in real time using W&B, which are displayed their online UI. In addition, the configuration variable values can also be logged to W&B and used to evaluate and compare experiments.

```zsh
python solve_ode/hydra_configs_wandb.py
```

## Scikit-Learn example

This example uses the Scikit-Learn machine learning library to train and evaluate models on supervised learning tasks. The configuration parameters are separated into the following groups: dataset, model, metric and preprocessing. The training and testing metric value is logged as well as a scatter plot of test predictions for regression tasks and a confusion matrix classification tasks.

To run with the default configurations:

```zsh
python sklearn_example/train.py
```

The default configurations can be orverriden in the commnad to train different models or change the task, for example:

```zsh
python train_sklearn.py dataset=iris task=classification model=svm +preprocessing=minmax metric=accuracy
```

W&B also support hyperparameter tuning using sweeps. To find the optimal hyperparameter for the random forest regressor using Bayesian optimisation, run:

```zsh
python sweep.py rforest --count=10
```

