# Dynamic GNN Quantization — Research Experiments

Preliminary experiments exploring quantization of Dynamic Graph Neural Network.


## Dataset

**WikiMaths** — a Discrete-Time Dynamic Graph (DTDG) benchmark from
PyTorch Geometric Temporal. Nodes = Wikipedia mathematics articles.
Edges = hyperlinks between articles over time. Task = node activity
forecasting accross snapshots

## Output Files

```
results/
  phase1_results.txt              # Baseline F1, AUC, time, memory
  baseline_loss_curve.png         # Training loss curve
  exp1_results.txt                # Quantization comparison table
  exp1_quantization_results.png   # F1 and model size bar charts
evolvegcn_baseline.pt             # Trained model weights
```

## Model Architecture

**EvolveGCN-O** (Pareja et al., AAAI 2020): a Dynamic GNN that evolves
the GCN weight matrix itself over time using a GRU cell. At each snapshot,
the GRU takes the previous weight matrix and outputs a new one — so the
model parameters change at every time step.

This makes it particularly interesting for quantization research because
quantization error introduced at one time step can propagate forward
through the GRU update mechanism.

## Key Concepts

- **Quantization**: reducing the numerical precision of model weights
  (e.g. FP32 → INT8 or INT4) to reduce memory and increase inference speed.
- **Post-training quantization (PTQ)**: applied after training, no retraining needed.
- **INT4 simulation**: PyTorch has no native INT4 support; we simulate it
  by rounding weights to 16 uniformly-spaced levels (2^4), the standard
  research technique used in papers like A2Q.
- **Snapshot contraction**: merging structurally similar consecutive snapshots
  to reduce temporal complexity before processing.
