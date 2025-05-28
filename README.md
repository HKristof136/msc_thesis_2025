# Project README

## Overview

This repository contains the codebase for my MSc thesis project. The project is organized into several folders, but this README will focus on the `pricer` and `neural_network` folders.

## Folder Structure

- `pricer/`
- `neural_network/`

## Pricer

The `pricer` folder contains the object-oriented implementation of the pricing algorithms used in the project. This includes various models and methods for pricing financial instruments.

### Key Files

- `analytical.py`: Contains the main pricing model implementation.
- `pde_solver_base.py`: Implementation of the Crank-Nicolson method for solving the Black-Scholes partial differential equation and the Craig-Sneyd ADI scheme for solving the Heston PDE.

### Usage

Each instrument has its own pricing Class initialized using a config dictionary, with methods price(), and to calculate the greeks delta(), vega(), ...
For the FDM method after initializing, the .solve() method can be used to solve the PDE on the grid from the config. The price() method here also requires a list of points to price with interpolation on the solved grid (linearly, or using splines).

The greeks are calculated using the analytical formulas in the Black-Scholes model for call and put, for the rest of the pricers numerical approximations are used.

## Neural Network

The `neural_network` folder contains the implementation of the neural network models pipeline, which contains instrument and model specific training and test data generation using multiple CPU cores, fully customisable config based model building using TensorFlow 2.0 Functional Model API or PyTorch neural networks, custom evaluation of a Pipeline and hyperparameter optimisation based on the evaluation.

### Key Files

- `torch_pipeline.py`: Defines the Pipeline Class, which includes data generation, building neural network architecture, and evaluation.
- `data_generator/data_generator_base.py`: Specific sample data generating functions for each instrument.
- `hyperparameter_optimization.py`: Random search hyperparameter optimisation using a custom parameter set of the config dictionary.

## Dependencies

Make sure to install the required dependencies before running the scripts. You can install them using:

```bash
pip install -r requirements.txt
```
