[![DOI](https://zenodo.org/badge/835600896.svg)](https://zenodo.org/doi/10.5281/zenodo.13382949)

```
# ~~~
# This file is part of the paper:
#
#           " Two-stage model reduction approaches for the efficient
#         and certified solution of parametrized optimal control problems "
#
#   https://github.com/HenKlei/REDUCED-OPT-CONTROL.git
#
# Copyright 2024 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Hendrik Kleikamp, Lukas Renelt
# ~~~
```

# Optimal control of parametrized linear systems using greedy and reduced basis algorithms
In this repository, we provide the code used for the numerical experiments in our paper "Two-stage model reduction
approaches for the efficient and certified solution of parametrized optimal control problems" by Hendrik Kleikamp and Lukas Renelt.

You find the preprint [here](https://arxiv.org/abs/tba).

## Installation
On a system with `git` (`sudo apt install git`), `python3` (`sudo apt install python3-dev`) and
`venv` (`sudo apt install python3-venv`) installed, the following commands should be sufficient
to install the `ml-control` package with all required dependencies in a new virtual environment:
```
git clone https://github.com/HenKlei/REDUCED-OPT-CONTROL.git
cd ml-control
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Running the experiments
To run the experiment shown in the paper, run the script [`cookies.py`](ml_control/examples/completely_reduced/cookies.py).
A smaller cookies example with less time steps that runs in about 15 minutes on a standard laptop is provided in
[`cookies_simplified.py`](ml_control/examples/completely_reduced/cookies_simplified.py).

## Questions
If you have any questions, feel free to contact us via email at <hendrik.kleikamp@uni-muenster.de>.
