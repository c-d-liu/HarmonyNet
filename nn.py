import numpy as np

class LemmaHopfieldNetwork:
    def __init__(self, pmi_df):
        """
        Initializes the Hopfield Network using a DataFrame of PMI scores.
        """
        # 1. Extract the unique vocabulary from the PMI pairs
        unique_words = set(pmi_df['word_x']).union(set(pmi_df['word_y']))
        self.vocab = sorted(list(unique_words))
        self.N = len(self.vocab)
        
        # 2. Create bidirectional mappings for fast lookups
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # 3. Initialize and populate the weight matrix
        print(f"Building {self.N}x{self.N} weight matrix...")
        self.W = np.zeros((self.N, self.N))
        
        for _, row in pmi_df.iterrows():
            i = self.word_to_idx[row['word_x']]
            j = self.word_to_idx[row['word_y']]
            weight = row['pmi']
            
            # Hopfield weights must be symmetric
            self.W[i, j] = weight
            self.W[j, i] = weight
            
        # Hopfield networks must have 0 on the diagonal (no self-connections)
        np.fill_diagonal(self.W, 0)

    def get_initial_state(self, active_words):
        """Creates a bipolar state array (-1, 1)."""
        state = np.full(self.N, -1.0)
        for w in active_words:
            if w in self.word_to_idx:
                state[self.word_to_idx[w]] = 1.0
            else:
                print(f"Warning: '{w}' not in vocabulary.")
        return state

    def retrieve(self, input_words, max_steps=10):
        """
        Runs the asynchronous update rule until the network converges 
        to a stable state, or hits max_steps.
        """
        state = self.get_initial_state(input_words)
        
        for step in range(max_steps):
            prev_state = state.copy()
            
            # Asynchronous update: evaluate nodes in a random order
            indices = np.random.permutation(self.N)
            for i in indices:
                # Calculate the weighted sum of inputs from all other nodes
                activation = np.dot(self.W[i], state)
                
                # Apply the sign function activation
                if activation > 0:
                    state[i] = 1.0
                elif activation < 0:
                    state[i] = -1.0
                # If activation == 0, the state remains unchanged
                    
            # Check if the network has settled into a stable state
            if np.array_equal(state, prev_state):
                print(f"Network converged in {step + 1} iterations.")
                break
        else:
            print(f"Stopped after {max_steps} iterations (no strict convergence).")
            
        # Decode the final state back into words
        retrieved_words = [
            self.idx_to_word[i] for i in range(self.N) if state[i] == 1.0
        ]
        return retrieved_words


# --- Example Usage ---
if __name__ == "__main__":
    import pandas as pd
    pmi_df = pd.read_csv('lemmatized_pmi_results.csv', sep='\t', encoding='utf-8')
    hopfield_net = LemmaHopfieldNetwork(pmi_df)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(hopfield_net.W, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.show()
    # 2. Provide a "stimulus" word to the network
    seed_words = ["american"]
    print(f"Input nodes activated: {seed_words}")

    # show the initial state of the network
    initial_state = hopfield_net.get_initial_state(seed_words)
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
    print(f"Final active state concepts: {associated_concepts}")