# TorchDendrite

## Table of Contents

- [Installation](#installation)
- [License](#license)

A PyTorch library for Spiking Dendrite-Enabled neurons. This library implements a hardware-aware Dendrite within
PyTorch, for use with the SNNTorch library or as a stand-alone PyTorch module.

## Overview

TorchDendrite is a module that implements a hardware-inspired dendritic cable function using Pytorch and SNNtorch. The
module contains trainable parameters and is compatible with both spiking and non-spiking neural networks.

## Getting started

### Installation

This project uses Hatch to manage the package and project. To install, make sure you are running Python >= 3.10.

Once you have cloned the repository, you can install it by navigating to the project directory and running:

```console
hatch install
 ```

### Examples and Notebooks

To get started with using TorchDendrite, you can refer to the examples and Jupyter notebooks provided in the notebooks
directory. These notebooks demonstrate how to use TorchDendrite to add dendrite layers to a PyTorch network and how to
use it with SNNtorch.

When training models with dendrites, it is recommended to use the Adam optimizer as it tends to perform better with
dendritic layers.


## License

`torchdendrite` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Project status

Currently in development; Examples and documentation are WIP.
