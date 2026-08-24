import pandas as pd
import numpy as np

from helper import load_words

def calculate_sentence_pmi(words_df, co_s_df):
    """
    Calculates Pointwise Mutual Information (PMI) for word pairs 
    based on their sentence co-occurrence frequencies.
    """
    # 1. Disregard the pre-calculated significance score
    co_s_df = co_s_df[['word1_id', 'word2_id', 'frequency']].copy()
    co_s_df = co_s_df.rename(columns={'frequency': 'f_xy'})
    
    # 2. Calculate N (Total number of tokens in the corpus)
    N = words_df['frequency'].sum()
    
    # 3. Merge to attach individual frequency and word string for Word 1
    pmi_df = co_s_df.merge(
        words_df[['word_id', 'word', 'frequency']], 
        left_on='word1_id', 
        right_on='word_id'
    ).rename(columns={'frequency': 'f_x', 'word': 'word_x'})
    
    # 4. Merge to attach individual frequency and word string for Word 2
    pmi_df = pmi_df.merge(
        words_df[['word_id', 'word', 'frequency']], 
        left_on='word2_id', 
        right_on='word_id'
    ).rename(columns={'frequency': 'f_y', 'word': 'word_y'})
    
    # 5. Apply the PMI formula
    # PMI = log2( (f(x,y) * N) / (f(x) * f(y)) )
    pmi_df['pmi'] = np.log2((pmi_df['f_xy'] * N) / (pmi_df['f_x'] * pmi_df['f_y']))
    
    # 6. Clean up the DataFrame for readability
    final_cols = ['word_x', 'word_y', 'f_xy', 'f_x', 'f_y', 'pmi']
    pmi_df = pmi_df[final_cols].sort_values(by='pmi', ascending=False)
    
    return pmi_df

if __name__ == "__main__":
    # Load the necessary data files
    from helper import load_words, load_co_s
    words_df = load_words('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-words.txt')
    co_s_df = load_co_s('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-co_s.txt')
    
    # Calculate PMI
    pmi_results = calculate_sentence_pmi(words_df, co_s_df)
    # Display the top 10 word pairs with the highest PMI
    print("Top 10 word pairs with the highest PMI:")
    print(pmi_results.head(10))
    print("Bottom 10 word pairs with the lowest PMI:")
    print(pmi_results.tail(10))