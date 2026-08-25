import pandas as pd
import numpy as np
import spacy
from collections import Counter
from itertools import combinations

# 1. Disable the heavy, unnecessary components ('parser' and 'ner')
# Keep 'tok2vec', 'tagger', and 'attribute_ruler' as the lemmatizer depends on them
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def generate_lemmatized_pmi_fast(sentences_df, min_co_occurrence=3):
    word_counts = Counter()
    co_occurrence_counts = Counter()
    total_tokens = 0
    
    # Drop empty rows and convert to a list of strings
    texts = sentences_df['sentence_text'].dropna().tolist()
    
    print(f"Processing {len(texts)} sentences...")
    
    # 2 & 3. Use nlp.pipe() for batching and multiprocessing
    # batch_size=1000 processes 1000 texts at once
    # n_process=-1 tells spaCy to use all available CPU cores
    for doc in nlp.pipe(texts, batch_size=1000, n_process=-1):
        
        valid_lemmas = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        
        total_tokens += len(valid_lemmas)
        word_counts.update(valid_lemmas)
        
        unique_lemmas = sorted(list(set(valid_lemmas)))
        co_occurrence_counts.update(combinations(unique_lemmas, 2))

    print("Calculations complete. Building DataFrame...")

    # (The rest of the DataFrame construction and PMI math remains exactly the same)
    co_s_records = [
        {'word_x': pair[0], 'word_y': pair[1], 'f_xy': count}
        for pair, count in co_occurrence_counts.items()
        if count >= min_co_occurrence 
    ]
    pmi_df = pd.DataFrame(co_s_records)
    
    words_dict = dict(word_counts)
    pmi_df['f_x'] = pmi_df['word_x'].map(words_dict)
    pmi_df['f_y'] = pmi_df['word_y'].map(words_dict)
    
    N = total_tokens
    pmi_df['pmi'] = np.log2((pmi_df['f_xy'] * N) / (pmi_df['f_x'] * pmi_df['f_y']))
    
    pmi_df = pmi_df.sort_values(by='pmi', ascending=False).reset_index(drop=True)
    
    return pmi_df, words_dict

# --- Example Usage ---
if __name__ == "__main__":
    # Load the sentences DataFrame
    from helper import load_sentences
    sentences_df = load_sentences('eng-simple_wikipedia_2021_300K/eng-simple_wikipedia_2021_300K-sentences.txt')
    
    # Generate lemmatized PMI
    pmi_results, word_frequencies = generate_lemmatized_pmi_fast(sentences_df, min_co_occurrence=3)
    
    # Display the top 10 word pairs with the highest PMI
    print("Top 10 word pairs with the highest PMI:")
    print(pmi_results.head(10))
    
    # Display the bottom 10 word pairs with the lowest PMI
    print("Bottom 10 word pairs with the lowest PMI:")
    print(pmi_results.tail(10))

    pmi_results.to_csv('lemmatized_pmi_results.csv', index=False, sep='\t', encoding='utf-8')
    word_freq_df = pd.DataFrame(list(word_frequencies.items()), columns=['word', 'frequency'])
    word_freq_df.to_csv('lemmatized_word_frequencies.csv', index=False, sep='\t', encoding='utf-8')
