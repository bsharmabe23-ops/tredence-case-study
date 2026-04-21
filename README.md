# Self-Pruning Neural Network
**Tredence AI Engineering Intern — Case Study Submission**

## Overview
A PyTorch implementation of a neural network that prunes itself during training
using learnable gate parameters and L1 sparsity regularization, trained on CIFAR-10.

## Files
- `pruning.py` — Complete implementation (PrunableLinear, training loop, evaluation)
- `report.md` — Analysis of results and λ trade-off
- `gate_distribution.png` — Gate value distribution plot

## How to Run
pip install torch torchvision matplotlib
python pruning.py
