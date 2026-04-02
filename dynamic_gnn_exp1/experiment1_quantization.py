"""
Experiment 1: Post-Training Quantization of Dynamic GNN

"""


import os
import sys
import copy
import time
import torch
import torch.nn.functional as F
import torch.quantization
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from sklearn.metrics import f1_score, roc_auc_score

os.makedirs("results", exist_ok=True)

# ── Guard: need baseline weights ──────────────────────────
if not os.path.exists("evolvegcn_baseline.pt"):
    print("ERROR: evolvegcn_baseline.pt not found.")
    print("Please run phase1_baseline_evolvegcn.py first.")
    sys.exit(1)

# ── 1. Reload Dataset ─────────────────────────────────────
print("="*55)
print("Loading WikiMaths dataset ...")
print("="*55)

loader  = WikiMathsDatasetLoader()
dataset = loader.get_dataset(lags=4)
_, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

sample        = next(iter(test_dataset))
node_features = sample.x.shape[1]

# ── 2. Model definition (must match Phase 1 exactly) ─────
class DynamicGNN(torch.nn.Module):
    def __init__(self, node_features, output_dim=1):
        super().__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear    = torch.nn.Linear(node_features, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        return self.linear(h)

# ── 3. Load trained weights ───────────────────────────────

base_model = DynamicGNN(node_features)
base_model.load_state_dict(torch.load("evolvegcn_baseline.pt", map_location='cpu'))
base_model.eval()
print("Loaded trained weights from evolvegcn_baseline.pt\n")

# ── 4. Helper: evaluate on CPU (needed for quantized models) ──
def evaluate(model, test_dataset, label):
    """
    Runs inference over all test snapshots.
    Quantized PyTorch models run on CPU — do not move to GPU here.
    """
    model.eval()
    all_preds, all_labels, latencies = [], [], []

    with torch.no_grad():
        for snapshot in test_dataset:
    
            x           = snapshot.x.cpu()
            edge_index  = snapshot.edge_index.cpu()
            edge_attr   = snapshot.edge_attr.cpu() if snapshot.edge_attr is not None else None

            t0    = time.perf_counter()
            y_hat = model(x, edge_index, edge_attr)
            latencies.append(time.perf_counter() - t0)

            all_preds.append(y_hat.squeeze().numpy())
            all_labels.append(snapshot.y.squeeze().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    threshold    = np.median(all_labels)
    pred_binary  = (all_preds  > threshold).astype(int)
    label_binary = (all_labels > threshold).astype(int)

    f1  = f1_score(label_binary, pred_binary, average='macro')
    try:
        auc = roc_auc_score(label_binary, all_preds)
    except Exception:
        auc = float('nan')

    avg_lat_ms = np.mean(latencies) * 1000

    print(f"  [{label}]")
    print(f"    Macro F1     : {f1:.4f}")
    print(f"    AUC          : {auc:.4f}")
    print(f"    Avg latency  : {avg_lat_ms:.2f} ms/snapshot")

    return {"label": label, "f1": f1, "auc": auc, "latency_ms": avg_lat_ms}

# ── 5. Helper: get model size ─────────────────────────────
def model_size_mb(model):
    path = "/tmp/_tmp_model.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024**2
    os.remove(path)
    return size

# ── STEP A: FP32 Baseline ─────────────────────────────────
print("="*55)
print("STEP A — FP32 baseline (no quantization)")
print("="*55)
fp32_model   = copy.deepcopy(base_model)
fp32_result  = evaluate(fp32_model, test_dataset, "FP32 baseline")
fp32_result["size_mb"] = model_size_mb(fp32_model)
print(f"    Model size   : {fp32_result['size_mb']:.3f} MB\n")

# ── STEP B: INT8 via PyTorch dynamic quantization ─────────
print("="*55)
print("STEP B — INT8 dynamic quantization")
print("  (Quantizes Linear + GRU/LSTM weights to 8-bit)")
print("="*55)

int8_model = copy.deepcopy(base_model)
int8_model = torch.quantization.quantize_dynamic(
    int8_model,
    {torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM},
    dtype=torch.qint8
)
int8_result  = evaluate(int8_model, test_dataset, "INT8 quantized")
int8_result["size_mb"] = model_size_mb(int8_model)
print(f"    Model size   : {int8_result['size_mb']:.3f} MB\n")

# ── STEP C: INT4 simulation ───────────────────────────────
print("="*55)
print("STEP C — INT4 simulated quantization")
print("  (Clamps each weight tensor to 16 levels — 2^4)")
print("  (Standard research technique; PyTorch has no native INT4)")
print("="*55)

def simulate_int4(model):

    m          = copy.deepcopy(model)
    NUM_LEVELS = 2 ** 4  # 16 levels for 4-bit

    with torch.no_grad():
        for name, param in m.named_parameters():
            if param.requires_grad and param.dim() > 1:
                mn    = param.data.min()
                mx    = param.data.max()
                scale = (mx - mn) / (NUM_LEVELS - 1)
                if scale == 0:
                    continue
                q     = torch.round((param.data - mn) / scale).clamp(0, NUM_LEVELS - 1)
                param.data = q * scale + mn
    return m

int4_model  = simulate_int4(base_model)
int4_result = evaluate(int4_model, test_dataset, "INT4 simulated")
# Theoretical size: 4/32 = 12.5% of FP32
int4_result["size_mb"] = fp32_result["size_mb"] * (4 / 32)
print(f"    Theoretical size : {int4_result['size_mb']:.3f} MB\n")

# ── Summary Table ─────────────────────────────────────────
f1_drop_int8  = fp32_result["f1"] - int8_result["f1"]
f1_drop_int4  = fp32_result["f1"] - int4_result["f1"]
size_save_int8 = (1 - int8_result["size_mb"] / fp32_result["size_mb"]) * 100
size_save_int4 = (1 - int4_result["size_mb"] / fp32_result["size_mb"]) * 100

print("="*65)
print("EXPERIMENT 1 — RESULTS SUMMARY")
print("="*65)
header = f"{'Precision':<16}{'F1 Score':<12}{'F1 Drop':<12}{'Size (MB)':<13}{'Latency (ms)'}"
print(header)
print("-"*65)
print(f"{'FP32':<16}{fp32_result['f1']:<12.4f}{'—':<12}{fp32_result['size_mb']:<13.3f}{fp32_result['latency_ms']:.2f}")
print(f"{'INT8':<16}{int8_result['f1']:<12.4f}{f1_drop_int8:<12.4f}{int8_result['size_mb']:<13.3f}{int8_result['latency_ms']:.2f}")
print(f"{'INT4 (sim)':<16}{int4_result['f1']:<12.4f}{f1_drop_int4:<12.4f}{int4_result['size_mb']:<13.3f}{'—'}")
print("="*65)

# ── Key observation ───────────────────────────────────────
print("\nKEY FINDING:")
if abs(f1_drop_int8) < 0.02:
    obs_int8 = "INT8 causes near-zero F1 drop — model is robust at 8-bit."
elif abs(f1_drop_int8) < 0.05:
    obs_int8 = "INT8 causes small but noticeable F1 drop."
else:
    obs_int8 = "INT8 causes significant F1 drop — temporal weights are precision-sensitive!"

if f1_drop_int4 > f1_drop_int8 + 0.02:
    obs_int4 = "INT4 is noticeably worse than INT8 — lower precision damages temporal memory more."
else:
    obs_int4 = "INT4 is similar to INT8 — the model is broadly robust to low precision here."

print(f"  {obs_int8}")
print(f"  {obs_int4}")

# ── Save results text ─────────────────────────────────────
lines = [
    "Experiment 1 — Quantization Results",
    "="*40,
    f"Dataset          : WikiMaths (DTDG benchmark)",
    f"Model            : EvolveGCN-O (trained in Phase 1)",
    "",
    f"{'Precision':<16}{'F1 Score':<12}{'F1 Drop':<12}{'Size (MB)':<13}{'Latency (ms)'}",
    "-"*65,
    f"{'FP32':<16}{fp32_result['f1']:<12.4f}{'—':<12}{fp32_result['size_mb']:<13.3f}{fp32_result['latency_ms']:.2f}",
    f"{'INT8':<16}{int8_result['f1']:<12.4f}{f1_drop_int8:<12.4f}{int8_result['size_mb']:<13.3f}{int8_result['latency_ms']:.2f}",
    f"{'INT4 (sim)':<16}{int4_result['f1']:<12.4f}{f1_drop_int4:<12.4f}{int4_result['size_mb']:<13.3f}—",
    "",
    "Key finding:",
    f"  {obs_int8}",
    f"  {obs_int4}",
]
with open("results/exp1_results.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\nResults saved → results/exp1_results.txt")

# ── Plot ──────────────────────────────────────────────────
labels  = ["FP32\n(baseline)", "INT8\n(quantized)", "INT4\n(simulated)"]
f1s     = [fp32_result["f1"],  int8_result["f1"],  int4_result["f1"]]
sizes   = [fp32_result["size_mb"], int8_result["size_mb"], int4_result["size_mb"]]
colors  = ["#185FA5", "#1D9E75", "#D85A30"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("EvolveGCN Quantization Experiment — Dynamic GNN (WikiMaths)", fontsize=12, fontweight='bold')

ax = axes[0]
bars = ax.bar(labels, f1s, color=colors, width=0.4, edgecolor='white')
ax.axhline(fp32_result["f1"], color='gray', linestyle='--', linewidth=0.9, label='FP32 reference')
ax.set_title("F1 Score vs. Quantization Level")
ax.set_ylabel("Macro F1 Score")
ax.set_ylim(max(0, min(f1s) - 0.08), min(1.0, max(f1s) + 0.06))
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.004, f"{v:.3f}",
            ha='center', fontsize=10, fontweight='bold')

ax = axes[1]
bars = ax.bar(labels, sizes, color=colors, width=0.4, edgecolor='white')
ax.set_title("Model Size (MB) vs. Quantization Level")
ax.set_ylabel("Size (MB)")
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.0002, f"{v:.3f}",
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("results/exp1_quantization_results.png", dpi=130)
print("Plot saved    → results/exp1_quantization_results.png")
print("\nExperiment 1 complete. Run the Git steps next.")
