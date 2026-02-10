import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Path to your text file
txt_path = "metrics_ce.txt"

# Lists to store parsed values
epochs = []
train_loss = []
val_loss = []

# Parse the text file line by line
with open(txt_path, "r") as f:
    for line in f:
        # Match the line using regex
        match = re.match(r"Epoch (\d+)/\d+ \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)", line)
        if match:
            epoch = int(match.group(1))
            t_loss = float(match.group(2))
            v_loss = float(match.group(3))
            epochs.append(epoch)
            train_loss.append(t_loss)
            val_loss.append(v_loss)

# Convert to DataFrame
df = pd.DataFrame({
    "epoch": epochs,
    "train_loss_epoch": train_loss,
    "val_loss": val_loss
})

# -----------------------------------------
# PLOT LOSS
# -----------------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_loss_epoch"], marker='o', label="Training Loss")
plt.plot(df["epoch"], df["val_loss"], marker='s', label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(range(1, 51), df_sorted["train_loss_epoch"], marker='o', label="Training Loss")
# plt.plot(range(1, 51), df_sorted["val_loss"], marker='s', label="Validation Loss")
# plt.title("Training and Validation Loss Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()