import numpy as np
from itertools import combinations

class SemanticHopfieldNetwork:
    # DO NOT USE
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
    
    def calculate_harmony(self, state):
        """
        Calculates the harmony (energy) of the network at any specific stage.
        Takes a bipolar state array (-1.0, 1.0) and uses the vectorized 
        matrix dot product formula.
        """
        # Vectorized Harmony calculation: 0.5 * (S^T * W * S)
        return 0.5 * np.dot(state, np.dot(self.W, state))

    def retrieve(self, input_words, max_steps=10, output_words=True, output_harmony=True, track_harmony=True):
        """
        Runs the standard bipolar update rule with clamped inputs.
        Optionally tracks and prints the harmony at every stage.
        """
        state, clamped_indices = self.get_initial_state(input_words)
        clamped_set = set(clamped_indices)
        
        # Track initial harmony before any updates occur
        if track_harmony:
            initial_harmony = self.calculate_harmony(state)
            print(f"Initial State Harmony: {initial_harmony:.4f}")
        
        for step in range(max_steps):
            prev_state = state.copy()
            # Randomize update order for sequential asynchronous updates
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
            
            # Calculate harmony at this exact stage of the update cycle
            if track_harmony:
                current_harmony = self.calculate_harmony(state)
                print(f"Step {step + 1} Harmony: {current_harmony:.4f}")
                    
            if np.array_equal(state, prev_state):
                print(f"Network converged in {step + 1} iterations.")
                break
        else:
            print(f"Stopped after {max_steps} iterations (no strict convergence).")
            
        retrieved_words = [
            self.idx_to_word[i] for i in range(self.N) if state[i] == 1.0
        ]

        if output_harmony:
            return self.calculate_harmony(state)
        elif output_words:
            return retrieved_words
        else:
            return state

    def incremental_retrieve(self, input_words, max_steps_per_word=10, output_harmony=True, output_words=True):
        """
        Incrementally clamps words from a sequence one by one, tracking the 
        corrected bipolar Delta Harmony (ΔH = 2 * h_k) when a node flips.
        """
        trajectory = []
        # Initialize an empty bipolar state and an empty clamped set
        state = np.full(self.N, -1.0)
        clamped_set = set()
        
        for word in input_words:
            if word not in self.word_to_idx:
                print(f"Warning: '{word}' not in vocabulary. Skipping.")
                trajectory.append({
                    'word': word,
                    'delta_harmony': 0.0,
                    'total_harmony': self.calculate_harmony(state),
                    'active_nodes': np.sum(state == 1.0)
                })
                continue
                
            idx = self.word_to_idx[word]
            clamped_set.add(idx)
            
            # 1. Calculate the local field (h_k) BEFORE flipping the node
            h_k = np.dot(self.W[idx], state)
            
            # 2. Apply the flip and calculate Delta Harmony
            if state[idx] != 1.0:
                delta_h = 2 * h_k  # The corrected bipolar delta harmony
                state[idx] = 1.0
                print(f"\n--- Clamped new word: '{word}' | ΔH: {delta_h:.4f} ---")
            else:
                delta_h = 0.0
                print(f"\n--- Clamped '{word}' (was already active) | ΔH: 0.0000 ---")
            
            # 3. Run the standard update loop until convergence for this stage
            for step in range(max_steps_per_word):
                prev_state = state.copy()
                indices = np.random.permutation(self.N)
                
                for i in indices:
                    if i in clamped_set:
                        continue
                        
                    activation = np.dot(self.W[i], state)
                    
                    if activation > 0:
                        state[i] = 1.0
                    elif activation < 0:
                        state[i] = -1.0
                        
                if np.array_equal(state, prev_state):
                    print(f"Network settled in {step + 1} iterations.")
                    break
            else:
                print(f"Stopped after {max_steps_per_word} iterations.")
            
            # 4. Display the total global harmony at this stable point
            current_harmony = 0.5 * np.dot(state, np.dot(self.W, state))
            active_count = np.sum(state == 1.0)
            #print(f"Total Harmony: {current_harmony:.4f} | Active Nodes: {active_count}")
            trajectory.append({
                'word': word,
                'delta_harmony': delta_h,
                'total_harmony': current_harmony,
                'active_nodes': active_count
            })
            
        # Decode the final state after all words are sequentially added
        retrieved_words = [
            self.idx_to_word[i] for i in range(self.N) if state[i] == 1.0
        ]

        if output_harmony:
            return trajectory
        elif output_words:
            return retrieved_words
        else:
            return state
        
class PMIHarmonyNetwork:
    def __init__(self, pmi_df):
        """
        Initializes the Harmony Network using raw PMI values.
        """
        # 1. Setup vocabulary
        unique_words = set(pmi_df['word_x']).union(set(pmi_df['word_y']))
        self.vocab = sorted(list(unique_words))
        self.N = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        # 2. Build the raw weight matrix
        print(f"Building {self.N}x{self.N} raw PMI weight matrix...")
        self.W = np.zeros((self.N, self.N))
        
        for _, row in pmi_df.iterrows():
            i = self.word_to_idx[row['word_x']]
            j = self.word_to_idx[row['word_y']]
            self.W[i, j] = row['pmi']
            self.W[j, i] = row['pmi']
            
        # Ensure the diagonal is 0 (no self-harmony)
        np.fill_diagonal(self.W, 0)

    def get_state(self, active_words):
        """
        Represents the network as a binary vector:
        1 for active input words, 0 for all other words.
        """
        state = np.zeros(self.N)
        valid_words = []
        
        for w in active_words:
            if w in self.word_to_idx:
                state[self.word_to_idx[w]] = 1.0
                valid_words.append(w)
            else:
                print(f"Warning: '{w}' not in vocabulary and will be ignored.")
                
        return state, valid_words

    def calculate_harmony(self, input_words):
        """
        Calculates the Harmony (total pairwise PMI) for a given set of words
        using the matrix dot product formula.
        """
        state, valid_words = self.get_state(input_words)
        
        # If fewer than 2 valid words, there are no pairs to sum
        if np.sum(state) < 2:
            return 0.0, valid_words
            
        # Vectorized Harmony calculation: 0.5 * (V^T * W * V)
        # np.dot(self.W, state) gives the weighted sum of inputs to each node
        # np.dot(state, ...) masks out the inactive nodes and sums the total
        harmony_score = 0.5 * np.dot(state, np.dot(self.W, state))
        
        return harmony_score, valid_words

    def calculate_harmony_explicit(self, input_words):
        """
        Alternative method: Calculates Harmony by explicitly iterating over pairs.
        Useful for debugging or if you want to see exactly which pairs contribute.
        """
        _, valid_words = self.get_state(input_words)
        total_harmony = 0.0
        
        for w1, w2 in combinations(valid_words, 2):
            idx1 = self.word_to_idx[w1]
            idx2 = self.word_to_idx[w2]
            total_harmony += self.W[idx1, idx2]
            
        return total_harmony, valid_words

import numpy as np

class IncrementalHarmonyNetwork:
    def __init__(self, pmi_df):
        # Setup vocabulary and matrix (same as before)
        unique_words = set(pmi_df['word_x']).union(set(pmi_df['word_y']))
        self.vocab = sorted(list(unique_words))
        self.N = len(self.vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        print(f"Building {self.N}x{self.N} raw PMI weight matrix...")
        self.W = np.zeros((self.N, self.N))
        for _, row in pmi_df.iterrows():
            i = self.word_to_idx[row['word_x']]
            j = self.word_to_idx[row['word_y']]
            self.W[i, j] = row['pmi']
            self.W[j, i] = row['pmi']
            
        np.fill_diagonal(self.W, 0)
        
        # Initialize the incremental state
        self.reset_state()

    def reset_state(self):
        """Clears the network to start a new sequence."""
        self.current_state = np.zeros(self.N)
        self.current_harmony = 0.0
        self.active_words = []

    def add_word(self, word):
        """
        Activates a single word and updates the harmony score incrementally.
        Returns the new total harmony and the delta (how much this word added).
        """
        if word not in self.word_to_idx:
            print(f"'{word}' not in vocabulary. Skipping.")
            return self.current_harmony, 0.0
            
        idx = self.word_to_idx[word]
        
        # If the word is already active, it adds nothing new to the harmony
        if self.current_state[idx] == 1.0:
            return self.current_harmony, 0.0
            
        # 1. Calculate the Delta Harmony (Vectorized)
        # Dot product of the new word's weight row against the active state vector
        delta_h = np.dot(self.W[idx], self.current_state)
        
        # 2. Update the network state
        self.current_state[idx] = 1.0
        self.current_harmony += delta_h
        self.active_words.append(word)
        
        return self.current_harmony, delta_h

    def process_sequence(self, sequence):
        """
        Feeds an ordered list of words into the network and tracks 
        the harmony trajectory.
        """
        trajectory = []
        for word in sequence:
            total_h, delta_h = self.add_word(word)
            trajectory.append({
                'word': word,
                'delta_harmony': delta_h,
                'total_harmony': total_h
            })
            
        return trajectory



# --- Example Usage ---
if __name__ == "__main__":
    import pandas as pd
    # load the PMI DataFrame
    pmi_df = pd.read_csv('wordlists/lemmatized_pmi_results_10K.csv', sep='\t', encoding='utf-8')
    pmi_df = pmi_df.astype({'word_x': str, 'word_y': str, 'f_xy': int, 'f_x': int, 'f_y': int, 'pmi': float})
    print(f"Mean PMI: {pmi_df['pmi'].mean():.4f}, Std Dev PMI: {pmi_df['pmi'].std():.4f}")
    # 1. Initialize the ZScoredHopfieldNetwork
    hopfield_net = ZScoredHopfieldNetwork(pmi_df)
    # 2. Provide a "stimulus" word to the network
    input_sequence = ['american', 'presidential', 'election', '2020']
    print("Processing input sequence with ZScoredHopfieldNetwork:")
    trajectory = hopfield_net.incremental_retrieve(input_sequence, max_steps_per_word=10, output_harmony=True, output_words=False)    
    for step in trajectory:
        print(f"Word: {step['word']}, Delta Harmony: {step['delta_harmony']:.4f}, Total Harmony: {step['total_harmony']:.4f}")

    # Control: just calculate the total harmony change by feeding the sequence as (a,) (a,b) (a,b,c) (a,b,c,d)
    prev_harmony = 0.0
    for i in range(1, len(input_sequence) + 1):
        sub_sequence = input_sequence[:i]
        total_h = hopfield_net.retrieve(sub_sequence, output_harmony=True, output_words=False, track_harmony=False)
        delta_h = total_h - prev_harmony
        prev_harmony = total_h
        print(f"Sub-sequence: {sub_sequence}, Total Harmony: {total_h:.4f}, Delta Harmony: {delta_h:.4f}")

    # test the incremental harmony network
    incremental_net = IncrementalHarmonyNetwork(pmi_df)
    print("\nProcessing input sequence with IncrementalHarmonyNetwork:")
    trajectory = incremental_net.process_sequence(input_sequence)
    for step in trajectory:
        print(f"Word: {step['word']}, Delta Harmony: {step['delta_harmony']:.4f}, Total Harmony: {step['total_harmony']:.4f}")