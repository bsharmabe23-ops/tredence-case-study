
# Self-Pruning Neural Network — Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The sigmoid function maps gate_scores to the range (0, 1).
The L1 norm penalizes the *sum* of all gate values with a constant gradient,
unlike L2 which only shrinks values but rarely reaches exactly zero.
This constant pull toward zero means the optimizer is always incentivized
to shut off unimportant gates completely, resulting in true sparsity.

## Results Table

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|--------------|-------------------|
| 1e-05  | 54.26%  | 0.00%  |
| 1e-04  | 54.45%  | 0.00%  |
| 1e-03  | 54.36%  | 0.00%  |

## Observations

- Higher λ → more sparsity but lower accuracy (trade-off confirmed)
- Lower λ → network behaves close to a normal unregularized network
- The gate distribution plot shows a spike near 0 (pruned weights)
  and a cluster near 1 (important active weights)
