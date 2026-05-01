import matplotlib.pyplot as plt

# Epochs
epochs = [1, 2, 3, 4, 5, 6]

# Train Loss
train_loss = [
    0.2797,
    0.2559,
    1.5671,  # spike due to corrupted image fallback (normal)
    0.2448,
    0.2426,
    0.2830
]

# Val Loss
val_loss = [
    0.2582,
    0.2559,
    0.2692,
    0.2617,
    0.2638,
    0.4325
]

# SRCC
srcc = [
    0.7163,
    0.7214,
    0.7222,
    0.7198,
    0.7177,
    0.7155
]

# PLCC
plcc = [
    0.7204,
    0.7285,
    0.7175,
    0.7220,
    0.7190,
    0.5606
]

# RMSE
rmse = [
    0.5081,
    0.5059,
    0.5188,
    0.5116,
    0.5136,
    0.6577
]

# -------------------------------
# Plot 1: Train vs Val Loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SigLIP → AVA Fine‑Tuning — Train vs Val Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 2: SRCC / PLCC
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, srcc, marker='o', label='SRCC')
plt.plot(epochs, plcc, marker='o', label='PLCC')
plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.title("SigLIP → AVA Fine‑Tuning — SRCC & PLCC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 3: RMSE
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, rmse, marker='o', color='red', label='RMSE')
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("SigLIP → AVA Fine‑Tuning — RMSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
