from nn import ZScoredHopfieldNetwork, IncrementalHarmonyNetwork
import pandas as pd
import spacy

sentence_csv = 'C:\\Users\\cliu\\PodCastECoG\\transcript_parsed_aligned.csv'
df = pd.read_csv(sentence_csv)
sentences = df['new_text'].dropna().tolist()
nlp = spacy.blank("en")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()

pmi_df = pd.read_csv('wordlists/lemmatized_pmi_results_10K_threshold_3_w_stopwords.csv', sep='\t', encoding='utf-8')

hopfield_traj = []
pmi_traj = []

for sentence in sentences[:10]:
    words = sentence.split()
    # lemmatize the words
    words = [word.lower().strip(".,!?;:()[]{}\"'") for word in words]
    words = [nlp(word)[0].lemma_ for word in words if word]  # filter out empty strings
    print(f"\nProcessing sentence: {sentence}")
    
    # Initialize the ZScoredHopfieldNetwork
    hopfield_net = ZScoredHopfieldNetwork(pmi_df)
    hopfield_trajectory = hopfield_net.incremental_retrieve(words, max_steps_per_word=10, output_harmony=True, output_words=False)
    
    # Initialize the IncrementalHarmonyNetwork
    incremental_net = IncrementalHarmonyNetwork(pmi_df)
    incremental_trajectory = incremental_net.process_sequence(words)

    if len(hopfield_trajectory) != len(incremental_trajectory):
        print(f"Warning: Trajectory lengths differ for sentence: {sentence}")
        print(f"ZScoredHopfieldNetwork trajectory length: {len(hopfield_trajectory)}")
        print(f"IncrementalHarmonyNetwork trajectory length: {len(incremental_trajectory)}")
        print("Hopfield trajectory:", hopfield_trajectory)
        print("Pairwise PMI trajectory:", incremental_trajectory)
        continue  # Skip this sentence if lengths differ

    hopfield_traj.extend(hopfield_trajectory)
    pmi_traj.extend(incremental_trajectory)

import pickle
# Save the trajectories to a pickle file for later analysis
with open('hopfield_trajectory.pkl', 'wb') as f:
    pickle.dump(hopfield_traj, f)
with open('pmi_trajectory.pkl', 'wb') as f:
    pickle.dump(pmi_traj, f)

import matplotlib.pyplot as plt

# Extract delta harmony values for plotting
hopfield_delta_harmony = [step['delta_harmony'] for step in hopfield_traj]
pmi_delta_harmony = [step['delta_harmony'] for step in pmi_traj]

# Plotting the delta harmony values for both networks
plt.figure(figsize=(6, 6))
plt.scatter(hopfield_delta_harmony, pmi_delta_harmony, alpha=0.5)
plt.title('Delta Harmony: Hopfield vs Pairwise PMI Network')
plt.xlabel('Hopfield Delta Harmony')
plt.ylabel('Pairwise PMI Delta Harmony')
vmin = min(min(hopfield_delta_harmony), min(pmi_delta_harmony))
vmax = max(max(hopfield_delta_harmony), max(pmi_delta_harmony))
plt.plot([vmin, vmax], [vmin, vmax], 'r--')  # Diagonal line for reference
plt.savefig('delta_harmony_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# plot histograms of delta harmony values for both networks
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(hopfield_delta_harmony, bins=100, alpha=0.7, label='Hopfield')
plt.title('Hopfield Delta Harmony')
plt.xlabel('Delta Harmony')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(pmi_delta_harmony, bins=100, alpha=0.7, label='Pairwise PMI', color='orange')
plt.title('Pairwise PMI Delta Harmony')
plt.xlabel('Delta Harmony')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('delta_harmony_histograms.png', dpi=300, bbox_inches='tight')
plt.show()