import json
import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=3):
    """
    Compute moving average of array 'a' with window size 'n'.
    """
    if len(a) < n:
        return np.array([])  # not enough data
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# Path to metrics.json
metrics_file = "./metrics.json"

# Load JSON lines safely
with open(metrics_file, "r") as f:
    metrics = [json.loads(line) for line in f if line.strip()]

# Extract train & val losses
train_loss = [float(v["loss_box_reg"]) for v in metrics if "loss_box_reg" in v]
val_loss = [float(v["val_loss_box_reg"]) for v in metrics if "val_loss_box_reg" in v]

# Moving average window
N = 40
train_loss_avg = moving_average(train_loss, n=N)
val_loss_avg = moving_average(val_loss, n=N)

# Plot only if data exists
plt.figure(figsize=(10, 6))

if train_loss_avg.size > 0:
    plt.plot(range(N - 1, len(train_loss)), train_loss_avg, label="train loss")

if val_loss_avg.size > 0:
    plt.plot(range(N - 1, len(val_loss)), val_loss_avg, label="val loss")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Box Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
