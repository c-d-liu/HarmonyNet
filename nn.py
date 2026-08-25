import numpy as np

class SemanticHopfieldNetwork:
    def __init__(self, pmi_df, threshold=None):
        # 1. Setup vocabulary and dictionaries (same as before)
        unique_words = set(pmi_df['word_x']).union(set(pmi_df['word_y']))
        self.vocab = sorted(list(unique_words))
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = pmi_df['pmi'].median()
        self.N = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # 2. Build weight matrix
        print(f"Building {self.N}x{self.N} weight matrix...")
        self.W = np.zeros((self.N, self.N))
        for _, row in pmi_df.iterrows():
            i = self.word_to_idx[row['word_x']]
            j = self.word_to_idx[row['word_y']]
            weight = row['pmi']
            self.W[i, j] = weight
            self.W[j, i] = weight
            
        np.fill_diagonal(self.W, 0)

    def get_initial_state(self, active_words):
        """Creates a binary state array (0, 1) instead of bipolar (-1, 1)."""
        state = np.zeros(self.N) # Changed from np.full(self.N, -1.0)
        input_indices = []
        for w in active_words:
            if w in self.word_to_idx:
                idx = self.word_to_idx[w]
                state[idx] = 1.0
                input_indices.append(idx)
        return state, input_indices

    def retrieve(self, input_words, max_steps=10, threshold=None):
        """
        Runs the update rule with Clamped Inputs and Binary States.
        """
        state, clamped_indices = self.get_initial_state(input_words)
        clamped_set = set(clamped_indices)
        
        for step in range(max_steps):
            prev_state = state.copy()
            indices = np.random.permutation(self.N)
            
            for i in indices:
                # Skip updating the clamped input nodes
                if i in clamped_set:
                    continue
                    
                # Calculate activation using only active (1.0) nodes
                activation = np.dot(self.W[i], state)
                
                # Binary thresholding
                if threshold is None:
                    threshold = self.threshold
                if activation > threshold:
                    state[i] = 1.0
                elif activation < threshold:
                    state[i] = 0.0
                    
            if np.array_equal(state, prev_state):
                print(f"Network converged in {step + 1} iterations.")
                break
        else:
            print(f"Stopped after {max_steps} iterations (no strict convergence).")
            
        retrieved_words = [
            self.idx_to_word[i] for i in range(self.N) if state[i] == 1.0
        ]
        return retrieved_words


class ZScoredHopfieldNetwork:
    def __init__(self, pmi_df):
        # 1. Setup vocabulary
        pmi_df['z_score'] = (pmi_df['pmi'] - pmi_df['pmi'].mean()) / pmi_df['pmi'].std()
        unique_words = set(pmi_df['word_x']).union(set(pmi_df['word_y']))
        self.vocab = sorted(list(unique_words))
        self.N = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # 2. Build the raw weight matrix
        print(f"Building {self.N}x{self.N} raw weight matrix...")
        self.W = np.zeros((self.N, self.N))
        for _, row in pmi_df.iterrows():
            i = self.word_to_idx[row['word_x']]
            j = self.word_to_idx[row['word_y']]
            self.W[i, j] = row['z_score']
            self.W[j, i] = row['z_score']
        
        # 4. Re-zero the diagonal to prevent self-connections
        np.fill_diagonal(self.W, 0)

    def get_initial_state(self, active_words):
        """Creates a bipolar state array (-1, 1)."""
        state = np.full(self.N, -1.0)
        clamped_indices = []
        for w in active_words:
            if w in self.word_to_idx:
                idx = self.word_to_idx[w]
                state[idx] = 1.0
                clamped_indices.append(idx)
        return state, clamped_indices

    def retrieve(self, input_words, max_steps=10, output_words=True):
        """
        Runs the standard bipolar update rule with clamped inputs.
        """
        state, clamped_indices = self.get_initial_state(input_words)
        clamped_set = set(clamped_indices)
        
        for step in range(max_steps):
            prev_state = state.copy()
            indices = np.random.permutation(self.N)
            
            for i in indices:
                # Skip clamped inputs
                if i in clamped_set:
                    continue
                    
                # Standard Hopfield activation
                activation = np.dot(self.W[i], state)
                
                # Bipolar thresholding strictly at 0
                if activation > 0:
                    state[i] = 1.0
                elif activation < 0:
                    state[i] = -1.0
                    
            if np.array_equal(state, prev_state):
                print(f"Network converged in {step + 1} iterations.")
                break
        else:
            print(f"Stopped after {max_steps} iterations (no strict convergence).")
            
        retrieved_words = [
            self.idx_to_word[i] for i in range(self.N) if state[i] == 1.0
        ]
        if output_words:
            return retrieved_words
        else:
            return state

# --- Example Usage ---
if __name__ == "__main__":
    import pandas as pd
    pmi_df = pd.read_csv('lemmatized_pmi_results.csv', sep='\t', encoding='utf-8')
    pmi_df = pmi_df.astype({'word_x': str, 'word_y': str, 'f_xy': int, 'f_x': int, 'f_y': int, 'pmi': float})
    print(f"Mean PMI: {pmi_df['pmi'].mean():.4f}, Std Dev PMI: {pmi_df['pmi'].std():.4f}")
    hopfield_net = ZScoredHopfieldNetwork(pmi_df)
    # 2. Provide a "stimulus" word to the network
    seed_words = ['american', 'president']  # Example input words
    print(f"Input nodes activated: {seed_words}")

    # show the initial state of the network
    initial_state, input_indices = hopfield_net.get_initial_state(seed_words)
    print(f"Initial active state: {initial_state}")

    # show the first update of the network
    first_update = initial_state.copy()
    indices = np.random.permutation(hopfield_net.N)
    for i in indices:
        activation = np.dot(hopfield_net.W[i], first_update)
        if activation > 0:
            first_update[i] = 1.0
        elif activation < 0:
            first_update[i] = -1.0
    print(f"State after first update: {first_update}")
    
    # 3. Let the network retrieve associated memories
    associated_concepts = hopfield_net.retrieve(seed_words)
    #print(f"Final active state concepts: {associated_concepts}")
    print(f"Number of associated concepts retrieved: {len(associated_concepts)}")
    print(f"Associated concepts retrieved: {associated_concepts[:50]} ...")
    print(f"{associated_concepts[-10:]}")
    print(f"Concepts inhibited (not retrieved): {sorted(list(set(hopfield_net.vocab) - set(associated_concepts))[:50])} ...")