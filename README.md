```
# ~~~
# This file is part of the paper:
#
#           " Be greedy and learn: efficient and certified algorithms
#                    for parametrized optimal control problems "
#
#   https://github.com/HenKlei/ml-control.git
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Hendrik Kleikamp, Martin Lazar, Cesare Molinari
# ~~~
```

# Optimal control of parametrized linear systems using machine learning
In this repository, we provide the code used for the numerical experiments in our paper "Be greedy and learn: efficient
and certified algorithms for parametrized optimal control problems" by Hendrik Kleikamp, Martin Lazar, and Cesare Molinari.

You find the preprint [here](https://arxiv.org/abs/tba).

## Installation
On a system with `git` (`sudo apt install git`), `python3` (`sudo apt install python3-dev`) and
`venv` (`sudo apt install python3-venv`) installed, the following commands should be sufficient
to install the `ml-control` package with all required dependencies in a new virtual environment:
```
git clone https://github.com/HenKlei/ml-control.git
cd ml-control
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Running the experiments
To reproduce the results, we provide the original scripts creating the results presented in
the paper in the directory [`ml_control/examples/`](ml_control/examples/).

To apply the greedy algorithm and the machine learning reduced models for the heat equation
example, run the script [`heat_equation_greedy_complex.py`](ml_control/examples/heat_equation/heat_equation_greedy_complex.py).
If you would like to create plots of optimal final time adjoints, optimal controls and states,
run the script [`heat_equation_plots_complex.py`](ml_control/examples/heat_equation/heat_equation_plots_complex.py).
We also provide different parametrizations and problem settings that are not contained in
the paper in the folder [`heat_equation/`](ml_control/examples/heat_equation/).

To apply the greedy algorithm and the machine learning reduced models for the damped wave
equation example, run the script [`damped_wave_equation_greedy.py`](ml_control/examples/wave_equation/damped_wave_equation_greedy.py).
If you would like to create plots of optimal final time adjoints, optimal controls and states,
run the script [`damped_wave_equation_plots.py`](mml_control/examples/wave_equation/damped_wave_equation_plots.py).
We also provide different parametrizations and problem settings that are not contained in
the paper in the folder [`wave_equation/`](ml_control/examples/wave_equation/).

## Questions
If you have any questions, feel free to contact us via email at <hendrik.kleikamp@uni-muenster.de>.
