# Neural Network Weight Pruning with Learnable Gates

A PyTorch implementation of **soft weight pruning** using learnable gate scores on CIFAR-10. The model learns *which weights to prune* automatically during training via L1 regularization on sigmoid gates.

---

## How It Works

Each weight in the network has a corresponding **gate score** (a learnable parameter). During the forward pass:

```
pruned_weight = weight × sigmoid(gate_score)
```

- Gate ≈ 1 → weight is kept  
- Gate ≈ 0 → weight is effectively pruned  

A **sparsity loss** (L1 penalty on gate values) is added to the classification loss, encouraging gates to push toward zero:

```
Total Loss = CrossEntropy Loss + λ × Σ sigmoid(gate_scores)
```

The trade-off between accuracy and sparsity is controlled by the **lambda (λ)** hyperparameter.

---

## Model Architecture

| Layer | Type             | Input → Output     |
|-------|------------------|--------------------|
| fc1   | PrunableLinear   | 3072 → 512         |
| fc2   | PrunableLinear   | 512 → 256          |
| fc3   | PrunableLinear   | 256 → 10           |

Input: CIFAR-10 images (3 × 32 × 32), flattened to 3072 features.  
Output: 10 class logits.

---

## Dataset

**CIFAR-10** — 60,000 color images across 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

- Training set: 50,000 images  
- Test set: 10,000 images  
- Batch size: 128

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity |
|------------|---------------|----------|
| 0.0001     | 51.94%        | 70.60%   |
| 0.001      | 48.37%        | 83.20%   |
| 0.01       | 39.57%        | 85.77%   |

**Key observation:** Higher λ → more aggressive pruning → higher sparsity but lower accuracy. The sweet spot here is **λ = 0.0001**, which achieves ~52% accuracy while still pruning over 70% of weights.

---

## Training Details

| Hyperparameter | Value  |
|----------------|--------|
| Optimizer      | Adam   |
| Learning Rate  | 0.001  |
| Epochs         | 22     |
| Loss Function  | CrossEntropyLoss + L1 sparsity penalty |

---

## Project Structure

```
.
├── pruning.py       # Main script (model, training, evaluation)
├── data/            # CIFAR-10 dataset (auto-downloaded)
└── README.md
```

---

## Requirements

```
torch
torchvision
matplotlib
numpy
```

Install with:

```bash
pip install torch torchvision matplotlib numpy
```

---

## Usage

Run the script directly. It trains three models with different λ values and prints a results table:

```bash
python pruning.py
```

For the `λ = 0.001` run, a **gate value distribution histogram** is also plotted to visualize how many weights have been pruned.

---

## Key Concepts

- **Soft Pruning**: Weights are not hard-removed; they're multiplied by a near-zero gate, which is differentiable and allows gradient-based learning.
- **L1 Regularization on Gates**: Encourages sparsity by penalizing non-zero gate values.
- **Sparsity Threshold**: A gate value < 0.01 is counted as "pruned" for reporting purposes.
