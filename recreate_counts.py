import pandas as pd
import numpy as np
import spacy
from collections import Counter
from itertools import combinations

# 1. Disable the heavy, unnecessary components ('parser' and 'ner')
# Keep 'tok2vec', 'tagger', and 'attribute_ruler' as the lemmatizer depends on them
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def generate_lemmatized_pmi_fast(sentences_df, min_co_occurrence=3, skip_stopwords=True):
    word_counts = Counter()      # token freq, for the frequency output file
    doc_counts = Counter()       # sentence freq, for PMI
    co_occurrence_counts = Counter()
    n_sentences = 0
    
    # Drop empty rows and convert to a list of strings
    texts = sentences_df['sentence_text'].dropna().tolist()
    
    print(f"Processing {len(texts)} sentences...")
    
    # 2 & 3. Use nlp.pipe() for batching and multiprocessing
    # batch_size=1000 processes 1000 texts at once
    # n_process=-1 tells spaCy to use all available CPU cores
    for doc in nlp.pipe(texts, batch_size=1000, n_process=-1):

        if skip_stopwords:
            valid_lemmas = [
                token.lemma_.lower() for token in doc 
                if not token.is_punct and not token.is_space and not token.is_stop
            ]
        else:
            valid_lemmas = [
                token.lemma_.lower() for token in doc 
                if not token.is_punct and not token.is_space
            ]
        
        word_counts.update(valid_lemmas)
        unique_lemmas = sorted(set(valid_lemmas))
        doc_counts.update(unique_lemmas)
        co_occurrence_counts.update(combinations(unique_lemmas, 2))
        n_sentences += 1

    print("Calculations complete. Building DataFrame...")

    # (The rest of the DataFrame construction and PMI math remains exactly the same)
    co_s_records = [
        {'word_x': pair[0], 'word_y': pair[1], 'f_xy': count}
        for pair, count in co_occurrence_counts.items()
        if count >= min_co_occurrence 
    ]
    pmi_df = pd.DataFrame(co_s_records)
    
    docs_dict = dict(doc_counts)
    pmi_df['f_x'] = pmi_df['word_x'].map(docs_dict)
    pmi_df['f_y'] = pmi_df['word_y'].map(docs_dict)
    pmi_df['pmi'] = np.log2((pmi_df['f_xy'] * n_sentences) / (pmi_df['f_x'] * pmi_df['f_y']))
    
    pmi_df = pmi_df.sort_values(by='pmi', ascending=False).reset_index(drop=True)
    
    return pmi_df, word_counts

# --- Example Usage ---
if __name__ == "__main__":
    # Load the sentences DataFrame
    from helper import load_sentences
    sentences_df = load_sentences('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-sentences.txt')
    
    # Generate lemmatized PMI
    pmi_results, word_frequencies = generate_lemmatized_pmi_fast(sentences_df, min_co_occurrence=0, skip_stopwords=False)
    
    # Display the top 10 word pairs with the highest PMI
    print("Top 10 word pairs with the highest PMI:")
    print(pmi_results.head(10))
    
    # Display the bottom 10 word pairs with the lowest PMI
    print("Bottom 10 word pairs with the lowest PMI:")
    print(pmi_results.tail(10))

    pmi_results.to_csv('wordlists/lemmatized_pmi_results_10K_threshold_0_w_stopwords.csv', index=False, sep='\t', encoding='utf-8')
    word_freq_df = pd.DataFrame(list(word_frequencies.items()), columns=['word', 'frequency'])
    word_freq_df.to_csv('wordlists/lemmatized_word_frequencies_10K_threshold_0_w_stopwords.csv', index=False, sep='\t', encoding='utf-8')
