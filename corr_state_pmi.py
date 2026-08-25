from nn import ZScoredHopfieldNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == "__main__":
    pmi_df = pd.read_csv('lemmatized_pmi_results_10K.csv', sep='\t', encoding='utf-8')
    print(f"Mean PMI: {pmi_df['pmi'].mean():.4f}, Std Dev PMI: {pmi_df['pmi'].std():.4f}")
    
    hopfield_net = ZScoredHopfieldNetwork(pmi_df)
    out_vec = []
    
    print("Simulating network convergence for all words...")
    for word in hopfield_net.vocab:
        seed_words = [word]
        # Assumes retrieve() is modified to return the raw state array when output_words=False
        retrieved_state = hopfield_net.retrieve(seed_words, max_steps=10, output_words=False)
        out_vec.append(retrieved_state)
        
    # N x N matrix of final output states
    out_mat = np.array(out_vec)
    
    # N x N Z-scored PMI weight matrix
    weight_mat = hopfield_net.W
    
    # =====================================================================
    # NEW: Correlate the Output State Matrix and the PMI Weight Matrix
    # =====================================================================
    
    # Flatten both matrices to 1D arrays for a global correlation calculation
    flat_out = out_mat.flatten()
    flat_weight = weight_mat.flatten()
    
    # Calculate Pearson correlation coefficient
    correlation, p_value = pearsonr(flat_out, flat_weight)
    print(f"\nGlobal Correlation (Output States vs PMI Weights): {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    # Visualize the correlation (using hexbin because scatter overlaps too much on -1 and 1)
    plt.figure(figsize=(10, 6))
    plt.hexbin(flat_weight, flat_out, gridsize=40, cmap='Blues', bins='log')
    plt.colorbar(label='log10(Count)')
    plt.title(f'PMI Weight vs Final Node State (Pearson r = {correlation:.4f})')
    plt.xlabel('Initial Z-Scored PMI Connection Weight')
    plt.ylabel('Final Node State (-1.0 or 1.0)')
    plt.savefig('weight_vs_state_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # save the output state matrix and weight matrix for further analysis
    np.save('output_state_matrix.npy', out_mat)
    np.save('weight_matrix.npy', weight_mat)