import pandas as pd
import numpy as np
import spacy
from collections import Counter
from itertools import combinations

# Ensure you have the model installed: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def generate_lemmatized_pmi(sentences_df, min_co_occurrence=3):
    """
    Tokenizes and lemmatizes sentences, counts frequencies, 
    and calculates PMI for word pairs.
    """
    word_counts = Counter()
    co_occurrence_counts = Counter()
    total_tokens = 0
    
    # Process each sentence
    for text in sentences_df['sentence_text'].dropna():
        # Let spaCy parse the sentence
        doc = nlp(text)
        
        # Extract valid lemmas: lowercased, ignoring punctuation and whitespaces
        valid_lemmas = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct and not token.is_space
        ]
        
        # 1. Update total token count and individual word frequencies (f_x)
        total_tokens += len(valid_lemmas)
        word_counts.update(valid_lemmas)
        
        # 2. Update sentence co-occurrences (f_xy)
        # Convert to a set to avoid counting (word, word) combinations 
        # or duplicate pairs within the exact same sentence
        unique_lemmas = sorted(list(set(valid_lemmas)))
        
        # Generate all unique pairs in this sentence and count them
        co_occurrence_counts.update(combinations(unique_lemmas, 2))

    # Convert co-occurrences into a DataFrame
    co_s_records = [
        {'word_x': pair[0], 'word_y': pair[1], 'f_xy': count}
        for pair, count in co_occurrence_counts.items()
        if count >= min_co_occurrence  # Filter out rare pairs to reduce noise
    ]
    pmi_df = pd.DataFrame(co_s_records)
    
    # Map individual word frequencies to the DataFrame
    words_dict = dict(word_counts)
    pmi_df['f_x'] = pmi_df['word_x'].map(words_dict)
    pmi_df['f_y'] = pmi_df['word_y'].map(words_dict)
    
    # Calculate PMI: log2( (f(x,y) * N) / (f(x) * f(y)) )
    N = total_tokens
    pmi_df['pmi'] = np.log2((pmi_df['f_xy'] * N) / (pmi_df['f_x'] * pmi_df['f_y']))
    
    # Sort by highest PMI score
    pmi_df = pmi_df.sort_values(by='pmi', ascending=False).reset_index(drop=True)
    
    return pmi_df, words_dict

# --- Example Usage ---
if __name__ == "__main__":
    # Load the sentences DataFrame
    from helper import load_sentences
    sentences_df = load_sentences('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-sentences.txt')
    
    # Generate lemmatized PMI
    pmi_results, word_frequencies = generate_lemmatized_pmi(sentences_df, min_co_occurrence=3)
    
    # Display the top 10 word pairs with the highest PMI
    print("Top 10 word pairs with the highest PMI:")
    print(pmi_results.head(10))
    
    # Display the bottom 10 word pairs with the lowest PMI
    print("Bottom 10 word pairs with the lowest PMI:")
    print(pmi_results.tail(10))

    pmi_results.to_csv('lemmatized_pmi_results.csv', index=False, sep='\t', encoding='utf-8')
    word_freq_df = pd.DataFrame(list(word_frequencies.items()), columns=['word', 'frequency'])
    word_freq_df.to_csv('lemmatized_word_frequencies.csv', index=False, sep='\t', encoding='utf-8')
