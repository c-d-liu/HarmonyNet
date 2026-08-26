import pickle
import numpy as np

# Load the trajectories from the pickle files
with open('hopfield_trajectory.pkl', 'rb') as f:
    hopfield_traj = pickle.load(f)

with open('pmi_trajectory.pkl', 'rb') as f:
    pmi_traj = pickle.load(f)

# plot only non-zero delta harmony values for both networks
hopfield_delta_harmony = [step['delta_harmony'] for step in hopfield_traj if step['delta_harmony'] != 0.0]
pmi_delta_harmony = [step['delta_harmony'] for step in pmi_traj if step['delta_harmony'] != 0.0]

# histogram of delta harmony values for both networks
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(hopfield_delta_harmony, bins=100, alpha=0.7, label='Hopfield')
plt.title('Hopfield Delta Harmony (Non-zero)')
plt.xlabel('Delta Harmony')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(pmi_delta_harmony, bins=100, alpha=0.7, label='Pairwise PMI', color='orange')
plt.title('Pairwise PMI Delta Harmony (Non-zero)')
plt.xlabel('Delta Harmony')
plt.ylabel('Frequency')
plt.savefig('delta_harmony_histograms_nonzero.png', dpi=300, bbox_inches='tight')
plt.show()

delta_harmony_both_positive = []
for hop_step, pmi_step in zip(hopfield_traj, pmi_traj):
    if hop_step['delta_harmony'] != 0 and pmi_step['delta_harmony'] != 0:
        delta_harmony_both_positive.append((hop_step['delta_harmony'], pmi_step['delta_harmony']))

# plot scatter of delta harmony values where both are positive
if delta_harmony_both_positive:
    hopfield_positive, pmi_positive = zip(*delta_harmony_both_positive)
    plt.figure(figsize=(6, 6))
    plt.scatter(hopfield_positive, pmi_positive, alpha=0.5)
    plt.title('Delta Harmony: Hopfield vs Pairwise PMI Network')
    plt.xlabel('Hopfield Delta Harmony')
    plt.ylabel('Pairwise PMI Delta Harmony')
    #vmin = min(min(hopfield_positive), min(pmi_positive))
    #vmax = max(max(hopfield_positive), max(pmi_positive))
    #plt.plot([vmin, vmax], [vmin, vmax], 'r--')  # Diagonal line for reference
    # linear regression line
    m, b = np.polyfit(hopfield_positive, pmi_positive, 1)
    plt.plot(hopfield_positive, m*np.array(hopfield_positive) + b, 'r--', label=f'Linear Fit: y={m:.2f}x+{b:.2f}')
    plt.savefig('delta_harmony_comparison_both_positive.png', dpi=300, bbox_inches='tight')
    plt.show()

from scipy.stats import pearsonr
# Calculate Pearson correlation for non-zero delta harmony values
pearson_corr, p_value = pearsonr(hopfield_positive, pmi_positive)
print(f"Pearson correlation (non-zero delta harmony): {pearson_corr:.4f}, P-value: {p_value:.4e}")