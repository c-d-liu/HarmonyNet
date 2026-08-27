from nn import ZScoredHopfieldNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import os
    import pickle
    if True:
        pmi_df = pd.read_csv('wordlists/lemmatized_pmi_results_10K_threshold_3_w_stopwords.csv', sep='\t', encoding='utf-8')
        print(f"Mean PMI: {pmi_df['pmi'].mean():.4f}, Std Dev PMI: {pmi_df['pmi'].std():.4f}")
        # check output of every single word in the vocabulary
        hopfield_net = ZScoredHopfieldNetwork(pmi_df)
        out_vec = []
        for word in hopfield_net.vocab:
            seed_words = [word]
            retrieved_words = hopfield_net.retrieve(seed_words, max_steps=10, output_harmony=False, output_words=False)
            out_vec.append(retrieved_words)
        # output pairwise correlation matrix
        out_mat = np.array(out_vec)
        corr_matrix = np.corrcoef(out_mat)
        import pickle
        with open('pairwise_correlation_matrix.pkl', 'wb') as f:
            pickle.dump(corr_matrix, f)
        with open('output_state_matrix.pkl', 'wb') as f:
            pickle.dump(out_mat, f)
    else:
        with open('pairwise_correlation_matrix.pkl', 'rb') as f:
            corr_matrix = pickle.load(f)
            print(f"Pairwise Correlation Matrix shape: {corr_matrix.shape}")
        with open('output_state_matrix.pkl', 'rb') as f:
            out_mat = pickle.load(f)
            print(f"Output State Matrix shape: {out_mat.shape}")
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Pairwise Correlation Matrix of Retrieved States')
    plt.xlabel('Word Index')
    plt.ylabel('Word Index')
    plt.savefig('pairwise_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    # do a clustering analysis on the correlation matrix
    from scipy.cluster.hierarchy import linkage, dendrogram
    linked = linkage(corr_matrix, method='ward')
    plt.figure(figsize=(12, 8))
    dendrogram(linked, labels=hopfield_net.vocab, leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram of Retrieved States')
    plt.xlabel('Word')
    plt.ylabel('Distance')
    plt.savefig('hierarchical_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()