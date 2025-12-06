import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# --- CONFIGURATION ---
log_filename = 'baseline_run_log2.txt'
output_filename = 'baseline_reproduction_plot2.png'
window_size = 50  # Smoothing window for the moving average

# --- 1. LOAD DATA FROM FILE ---
print(f"Reading logs from: {log_filename}...")

if not os.path.exists(log_filename):
    print(f"Error: File '{log_filename}' not found!")
    print("   -> Please save your terminal output to this text file first.")
    exit()

with open(log_filename, "r") as f:
    log_data = f.read()

# --- 2. PARSE DATA ---

pattern = r"Episode (\d+)\s+Total Reward: ([\d\.\-]+)\s+Steps: (\d+)"
#pattern = r"Episode (\d+) \| Reward: ([\d\.\-]+) \| Steps: (\d+)"
matches = re.findall(pattern, log_data)

if not matches:
    print("No matching data found in the file. Check the format!")
    exit()

print(f"Found {len(matches)} episodes. Processing...")

# Convert to DataFrame
df = pd.DataFrame(matches, columns=['Episode', 'Reward', 'Steps'])
df['Episode'] = df['Episode'].astype(int)
df['Reward'] = df['Reward'].astype(float)
df['Steps'] = df['Steps'].astype(int)

# --- 3. CALCULATE MOVING AVERAGES ---
df['Reward_MA'] = df['Reward'].rolling(window=window_size).mean()
df['Steps_MA'] = df['Steps'].rolling(window=window_size).mean()

# --- 4. PLOT ---
plt.figure(figsize=(14, 6))

# Subplot 1: Rewards
plt.subplot(1, 2, 1)
plt.plot(df['Episode'], df['Reward'], alpha=0.2, color='gray', label='Raw Reward')
plt.plot(df['Episode'], df['Reward_MA'], color='blue', linewidth=2, label=f'Moving Avg ({window_size})')
plt.title('Training Curve: Reward Stability')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Steps
plt.subplot(1, 2, 2)
plt.plot(df['Episode'], df['Steps'], alpha=0.3, color='orange', label='Raw Steps')
plt.plot(df['Episode'], df['Steps_MA'], color='red', linewidth=2, label=f'Moving Avg ({window_size})')
plt.title('Agent Behavior: Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid(True, alpha=0.3)

# Save and Show
plt.tight_layout()
plt.savefig(output_filename)
print(f"Success! Plot saved as '{output_filename}'")
plt.show()