"""
============================================================
Phase 1: Baseline Dynamic GNN Training — EvolveGCN-O

"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from sklearn.metrics import f1_score, roc_auc_score

os.makedirs("results", exist_ok=True)

# ── 1. Dataset ────────────────────────────────────────────
print("="*55)
print("Loading WikiMaths dynamic graph dataset...")
print("(Auto-downloads ~2 MB on first run — please wait)")
print("="*55)

loader  = WikiMathsDatasetLoader()
dataset = loader.get_dataset(lags=4)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print(f"Training snapshots : {train_dataset.snapshot_count}")
print(f"Test snapshots     : {test_dataset.snapshot_count}")

sample        = next(iter(train_dataset))
node_features = sample.x.shape[1]
print(f"Node features/snap : {node_features}")

# ── 2. Model ──────────────────────────────────────────────
class DynamicGNN(torch.nn.Module):
    def __init__(self, node_features, output_dim=1):
        super().__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear    = torch.nn.Linear(node_features, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        return self.linear(h)

# ── 3. Setup ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice : {device}")
if device.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

model     = DynamicGNN(node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
EPOCHS    = 200

print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")



# ── 4. Train ──────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs ...")
print("─"*45)

train_losses = []
peak_mem_mb  = 0.0
t_start      = time.time()

for epoch in range(EPOCHS):
    model.train()

    # Reset temporal state at the start of each epoch
    model.recurrent.reinitialize_weight()

    optimizer.zero_grad()

    epoch_loss = 0.0
    steps = 0
    total_loss = 0.0

    for snapshot in train_dataset:
        snapshot = snapshot.to(device)

        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss  = criterion(y_hat.squeeze(), snapshot.y.squeeze())
        
        total_loss = total_loss + loss
        epoch_loss += loss.item()
        steps += 1

    # Average loss for logging + stable gradients
    avg_loss = epoch_loss / steps
    train_losses.append(avg_loss)

    total_loss = total_loss / steps
    total_loss.backward()

  
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated(0) / 1024**2
        peak_mem_mb = max(peak_mem_mb, mem)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  |  Loss: {avg_loss:.4f}")

train_time = time.time() - t_start
print("─"*45)
print(f"Training complete in {train_time:.1f} s")

# ── 5. Evaluate ───────────────────────────────────────────
model.eval()
model.recurrent.reinitialize_weight()

all_preds, all_labels = [], []

with torch.no_grad():
    for snapshot in test_dataset:
        snapshot = snapshot.to(device)
        y_hat    = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        all_preds.append(y_hat.squeeze().cpu().numpy())
        all_labels.append(snapshot.y.squeeze().cpu().numpy())

all_preds    = np.concatenate(all_preds)
all_labels   = np.concatenate(all_labels)
threshold    = np.median(all_labels)
pred_binary  = (all_preds  > threshold).astype(int)
label_binary = (all_labels > threshold).astype(int)

f1 = f1_score(label_binary, pred_binary, average='macro')
try:
    auc = roc_auc_score(label_binary, all_preds)
except Exception:
    auc = float('nan')

# ── 6. Print ──────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

print("\n" + "="*45)
print("BASELINE RESULTS  (FP32 — Full Precision)")
print("="*45)
print(f"  Macro F1 Score  : {f1:.4f}")
print(f"  AUC Score       : {auc:.4f}")
print(f"  Training Time   : {train_time:.1f} s")
print(f"  Peak GPU Memory : {peak_mem_mb:.1f} MB")
print("="*45)
print("\n>>> SAVE THESE NUMBERS — compare them against Experiment 1!")

# ── 7. Save model + results ───────────────────────────────
torch.save(model.state_dict(), "evolvegcn_baseline.pt")

result_lines = [
    "Phase 1 — Baseline Results (FP32)",
    "="*38,
    f"Dataset           : WikiMaths (DTDG benchmark)",
    f"Model             : EvolveGCN-O",
    f"Device            : {device} ({gpu_name})",
    f"Epochs            : {EPOCHS}",
    f"Macro F1 Score    : {f1:.4f}",
    f"AUC Score         : {auc:.4f}",
    f"Training Time     : {train_time:.1f} s",
    f"Peak GPU Memory   : {peak_mem_mb:.1f} MB",
]
with open("results/phase1_results.txt", "w") as fh:
    fh.write("\n".join(result_lines) + "\n")

print("\nSaved:")
print("  evolvegcn_baseline.pt")
print("  results/phase1_results.txt")

# ── 8. Plot ───────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_losses, color='#185FA5', linewidth=1.6)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("EvolveGCN-O — Training Loss Curve (FP32 Baseline)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/baseline_loss_curve.png", dpi=130)
print("  results/baseline_loss_curve.png")
print("\nAll done. Run experiment1_quantization.py next.")
