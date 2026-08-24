import pandas as pd

def load_sentences(filepath):
    """
    Loads the sentences text file.
    Contains the core text of the corpus mapping a unique ID to the actual sentence.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['sentence_id', 'sentence_text']
    )

def load_words(filepath):
    """
    Loads the vocabulary file.
    Contains all unique words in the corpus, assigning each a unique ID and providing 
    its total frequency across all sentences.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['word_id', 'word', 'frequency']
    )

def load_sources(filepath):
    """
    Loads the sources file.
    Contains the provenance of the text, mapping a source ID to the original 
    URL and the date it was crawled.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['source_id', 'url', 'date_crawled']
    )

def load_inv_so(filepath):
    """
    Loads the sentence-to-source mapping.
    An inverted index that links every sentence ID to the source ID it was scraped from.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['sentence_id', 'source_id']
    )

def load_inv_w(filepath):
    """
    Loads the word-to-sentence mapping.
    An inverted index that links a word ID to the sentence ID it appears in, 
    along with its numerical position index within that specific sentence.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['word_id', 'sentence_id', 'position_index']
    )

def load_co_s(filepath):
    """
    Loads the sentence co-occurrence file.
    Contains statistical data on which pairs of words frequently appear together 
    in the same sentence, including their joint frequency and a significance score.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['word1_id', 'word2_id', 'frequency', 'significance_score']
    )

def load_co_n(filepath):
    """
    Loads the neighbor co-occurrence file.
    Contains statistical data on which pairs of words frequently appear as 
    direct adjacent neighbors, including their joint frequency and a significance score.
    """
    return pd.read_csv(
        filepath, 
        sep='\t', 
        header=None, 
        encoding='utf-8',
        names=['word1_id', 'word2_id', 'frequency', 'significance_score']
    )

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these paths with the actual locations of your files
    # df_sentences = load_sentences('eng-simple_wikipedia_2021_10K-sentences.txt')
    # df_words = load_words('eng-simple_wikipedia_2021_10K-words.txt')
    
    # Example: Print the first 5 rows of the sentences dataframe
    # print(df_sentences.head())
    loaded_sentences = load_sentences('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-sentences.txt')
    print(loaded_sentences.head())
    loaded_words = load_words('eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-words.txt')
    print(loaded_words.head())