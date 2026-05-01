import matplotlib.pyplot as plt

# Your logged values
epochs = list(range(1, 11))

train_loss = [
    1.7706, 1.6553, 1.5974, 1.5266, 1.4411,
    1.3366, 1.2449, 1.1398, 1.0546, 0.9739
]

val_loss = [
    1.6331, 1.5987, 1.6020, 1.6153, 1.6137,
    1.6340, 1.6434, 1.6844, 1.7089, 1.7167
]

srcc = [
    0.4620, 0.4704, 0.4678, 0.4747, 0.4722,
    0.4577, 0.4542, 0.4416, 0.4291, 0.4312
]

plcc = [
    0.4915, 0.4959, 0.4959, 0.4983, 0.4963,
    0.4832, 0.4809, 0.4653, 0.4546, 0.4544
]

rmse = [
    1.2779, 1.2644, 1.2657, 1.2709, 1.2703,
    1.2782, 1.2819, 1.2978, 1.3072, 1.3102
]

# -------------------------------
# Plot 1: Train vs Val Loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SigLIP TAD66K — Train vs Val Loss")
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
plt.title("SigLIP TAD66K — SRCC & PLCC")
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
plt.title("SigLIP TAD66K — RMSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
