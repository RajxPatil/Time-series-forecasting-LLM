# scripts/plot_enhanced_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# Load log
df = pd.read_csv("train_log.csv", parse_dates=["timestamp"])
print("Loaded:", len(df), "rows")

sns.set_theme(style="whitegrid", font_scale=1.2)

# === 1️⃣ Smooth Train vs Val Loss with gradient fill ===
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_loss"], color="#1f77b4", lw=2, marker="o", label="Train Loss")
plt.fill_between(df["epoch"], df["train_loss"], alpha=0.2, color="#1f77b4")
plt.plot(df["epoch"], df["val_loss"], color="#ff7f0e", lw=2, marker="s", label="Validation Loss")
plt.fill_between(df["epoch"], df["val_loss"], alpha=0.2, color="#ff7f0e")
plt.title("Training vs Validation Loss (MSE)", fontsize=15, weight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figures/loss_curve_fancy.png", dpi=300)
print("✅ Saved: figures/loss_curve_fancy.png")

# === 2️⃣ Combined subplot for all metrics ===
fig, axs = plt.subplots(1, 3, figsize=(14,4))

axs[0].plot(df["epoch"], df["train_loss"], "o-", label="Train", color="steelblue")
axs[0].plot(df["epoch"], df["val_loss"], "s--", label="Val", color="orange")
axs[0].set_title("MSE Loss")
axs[0].set_xlabel("Epoch")

axs[1].plot(df["epoch"], df["test_loss"], "d-", color="seagreen", label="Test Loss")
axs[1].set_title("Test Loss")

axs[2].plot(df["epoch"], df["mae_loss"], "p-", color="crimson", label="MAE")
axs[2].set_title("MAE")

for ax in axs:
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")

plt.suptitle("Model Performance Metrics Over Epochs", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/combined_metrics.png", dpi=300)
print("✅ Saved: figures/combined_metrics.png")

# === 3️⃣ Epoch runtime analysis ===
plt.figure(figsize=(7,4))
plt.bar(df["epoch"], df["epoch_time_s"], color="mediumorchid", alpha=0.8)
plt.title("Training Time per Epoch", fontsize=14, weight="bold")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
for i, t in enumerate(df["epoch_time_s"]):
    plt.text(df["epoch"][i], t + 1, f"{t:.1f}s", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("figures/epoch_runtime.png", dpi=300)
print("✅ Saved: figures/epoch_runtime.png")

# === 4️⃣ Radar chart for normalized metrics ===
from math import pi

metrics = ["train_loss", "val_loss", "test_loss", "mae_loss"]

# Convert to numeric safely
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

# Drop rows with NaN in these columns
df = df.dropna(subset=metrics)

# Normalize final-epoch metrics (last row / max)
normed = df[metrics].iloc[-1] / df[metrics].max()

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
values = normed.tolist()
values += values[:1]
ax.plot(angles, values, "o-", linewidth=2, label="Normalized Final Metrics")
ax.fill(angles, values, alpha=0.25, color="dodgerblue")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title("Normalized Performance Radar Chart", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("figures/radar_metrics.png", dpi=300)
print("✅ Saved: figures/radar_metrics.png")
