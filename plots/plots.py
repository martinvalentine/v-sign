import re
import matplotlib.pyplot as plt

# Path to the log file
log_path = '/home/martinvalentine/Desktop/v-sign/outputs/logs/baseline_res18/log.txt'

# Lists to store epochs and WERs
epochs = []
wers = []

# Regular expression to match Dev WER lines
wer_pattern = re.compile(r'Dev WER: ([0-9.]+)%')
epoch_pattern = re.compile(r'Epoch (\d+)')

current_epoch = -1

with open(log_path, 'r') as f:
    for line in f:
        # Update current epoch if found
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        # Find Dev WER and associate with current epoch
        wer_match = wer_pattern.search(line)
        if wer_match:
            wer = float(wer_match.group(1))
            # Try to get the epoch from the previous 'Best_dev' line if available
            epochs.append(current_epoch)
            wers.append(wer)

# Print start, lowest, and final WER
if wers:
    start_wer = wers[0]
    start_epoch = epochs[0]
    final_wer = wers[-1]
    final_epoch = epochs[-1]
    min_wer = min(wers)
    min_index = wers.index(min_wer)
    min_epoch = epochs[min_index]
    print(f"Start WER: {start_wer:.2f}% (Epoch {start_epoch})")
    print(f"Lowest WER: {min_wer:.2f}% (Epoch {min_epoch})")
    print(f"Final WER: {final_wer:.2f}% (Epoch {final_epoch})")
else:
    print("No WER data found in log file.")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, wers, marker='o', label='Dev WER')

# Highlight start, lowest, and final WER
if wers:
    # Start
    plt.scatter(epochs[0], wers[0], color='green', s=100, zorder=5, label=f'Start ({wers[0]:.2f}%)')
    plt.annotate(f'Start\n{wers[0]:.2f}%', (epochs[0], wers[0]), textcoords="offset points", xytext=(-30,10), ha='center', color='green')
    # Lowest
    min_wer = min(wers)
    min_index = wers.index(min_wer)
    plt.scatter(epochs[min_index], min_wer, color='red', s=100, zorder=5, label=f'Lowest ({min_wer:.2f}%)')
    plt.annotate(f'Lowest\n{min_wer:.2f}%', (epochs[min_index], min_wer), textcoords="offset points", xytext=(0,-30), ha='center', color='red')
    # Final
    plt.scatter(epochs[-1], wers[-1], color='blue', s=100, zorder=5, label=f'Final ({wers[-1]:.2f}%)')
    plt.annotate(f'Final\n{wers[-1]:.2f}%', (epochs[-1], wers[-1]), textcoords="offset points", xytext=(30,10), ha='center', color='blue')

plt.xlabel('Epoch')
plt.ylabel('Dev WER (%)')
plt.title('Training Progress: Dev WER vs. Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()