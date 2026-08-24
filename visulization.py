import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nn import LemmaHopfieldNetwork

def retrieve_and_visualize(hopfield_net, input_words, max_steps=10):
    """
    Runs the sequential update rule and plots a heatmap of the state changes.
    Assumes 'hopfield_net' is an instance of the LemmaHopfieldNetwork.
    """
    # Initialize state
    state = hopfield_net.get_initial_state(input_words)
    
    # Track the state history (Step 0 is the initial input)
    history = [state.copy()]
    
    # 1. Run the Sequential Updates
    for step in range(max_steps):
        prev_state = state.copy()
        
        # Asynchronous update loop
        indices = np.random.permutation(hopfield_net.N)
        for i in indices:
            activation = np.dot(hopfield_net.W[i], state)
            if activation > 0:
                state[i] = 1.0
            elif activation < 0:
                state[i] = -1.0
                
        history.append(state.copy())
        
        if np.array_equal(state, prev_state):
            print(f"Network converged in {step + 1} iterations.")
            break
            
    # 2. Process Data for Visualization
    # Convert history to a matrix of Shape: (Num_Nodes, Num_Steps)
    history_matrix = np.array(history).T 
    
    # Filter for readability: Only keep nodes that were active (1.0) at some point
    # We ignore nodes that stayed at -1.0 the entire time
    active_mask = np.any(history_matrix == 1.0, axis=1)
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) == 0:
        print("No nodes were active. Nothing to visualize.")
        return
        
    filtered_matrix = history_matrix[active_indices]
    filtered_words = [hopfield_net.idx_to_word[i] for i in active_indices]
    
    # 3. Draw the Heatmap
    # Dynamically scale the figure height based on how many words are active
    fig_height = max(4, len(filtered_words) * 0.4)
    plt.figure(figsize=(10, fig_height))
    
    # Plot using Seaborn
    ax = sns.heatmap(
        filtered_matrix, 
        cmap="coolwarm",       # Blue for -1, Red for 1
        cbar_kws={'ticks': [-1, 1], 'label': 'Node State'},
        yticklabels=filtered_words,
        xticklabels=range(len(history)),
        linewidths=0.5, 
        linecolor='black'
    )
    
    # Formatting
    plt.title("Hopfield Network Sequential Update Trajectory", pad=15)
    plt.xlabel("Update Epoch (0 = Initial Input)")
    plt.ylabel("Active Words")
    
    # Adjust layout so long words aren't cut off
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    import pandas as pd
    # Load the PMI results and initialize the Hopfield network
    pmi_df = pd.read_csv('lemmatized_pmi_results.csv', sep='\t', encoding='utf-8')
    hopfield_net = LemmaHopfieldNetwork(pmi_df)
    
    # Provide a "stimulus" word to the network
    seed_words = ["algorithm", "data"]
    print(f"Input nodes activated: {seed_words}")
    
    # Run the retrieval and visualize the state changes
    retrieve_and_visualize(hopfield_net, seed_words, max_steps=10)