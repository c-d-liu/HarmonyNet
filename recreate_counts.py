import numpy as np
import pandas as pd
import scipy.sparse as sp
import spacy
from array import array
from collections import Counter

# --- Pipeline setup -----------------------------------------------------------
# FAST_LEMMAS=True skips the neural tok2vec+tagger entirely and uses the
# lookup lemmatizer instead. Roughly an order of magnitude faster, at the cost
# of POS-sensitive lemmas ("saw" -> "saw" not "see", "leaves" -> "leaf" always).
# Requires: pip install spacy-lookups-data
FAST_LEMMAS = True

if FAST_LEMMAS:
    nlp = spacy.blank("en")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()
else:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def generate_lemmatized_pmi_fast(
    sentences_df,
    min_co_occurrence=3,
    skip_stopwords=True,
    batch_size=1000,
    n_process=1,
):
    """Compute sentence-level PMI over lemmas.

    Returns (pmi_df, token_frequencies).

    Note on counts: f_xy, f_x and f_y are all *sentence* frequencies (number of
    sentences containing the term), and N is the number of sentences. All three
    probabilities must be over the same sample space for PMI to be meaningful.
    The returned token_frequencies dict is raw token counts, kept separately
    because it is useful output but is NOT what feeds the PMI formula.
    """
    texts = sentences_df["sentence_text"].dropna().tolist()
    print(f"Processing {len(texts)} sentences...")

    vocab = {}                  # lemma -> integer id
    token_counts = Counter()    # raw token frequency, for the output file
    rows = array("i")           # sentence index
    cols = array("i")           # word index
    n_sentences = 0

    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        if skip_stopwords:
            lemmas = [
                t.lemma_.lower()
                for t in doc
                if not t.is_punct and not t.is_space and not t.is_stop
            ]
        else:
            lemmas = [
                t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space
            ]

        token_counts.update(lemmas)

        for lemma in set(lemmas):
            idx = vocab.get(lemma)
            if idx is None:
                idx = len(vocab)
                vocab[lemma] = idx
            rows.append(n_sentences)
            cols.append(idx)

        n_sentences += 1

    print(f"Vocabulary: {len(vocab)} lemmas. Building co-occurrence matrix...")

    # Binary sentence x word incidence matrix.
    X = sp.csr_matrix(
        (
            np.ones(len(rows), dtype=np.int32),
            (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32)),
        ),
        shape=(n_sentences, len(vocab)),
        dtype=np.int32,
    )

    # Sentence frequency per word == column sums of the binary matrix.
    doc_freq = np.asarray(X.sum(axis=0)).ravel()

    # One matmul replaces the entire per-sentence combinations() loop.
    # k=1 takes the strict upper triangle: each unordered pair once, no diagonal.
    C = sp.triu(X.T @ X, k=1, format="coo")

    # Filter before materialising anything in pandas.
    keep = C.data >= min_co_occurrence if min_co_occurrence > 0 else slice(None)
    word_x_idx = C.row[keep]
    word_y_idx = C.col[keep]
    f_xy = C.data[keep].astype(np.float64)

    print(f"{len(f_xy)} pairs retained. Computing PMI...")

    f_x = doc_freq[word_x_idx].astype(np.float64)
    f_y = doc_freq[word_y_idx].astype(np.float64)

    pmi = np.log2((f_xy * n_sentences) / (f_x * f_y))

    id_to_lemma = np.empty(len(vocab), dtype=object)
    for lemma, idx in vocab.items():
        id_to_lemma[idx] = lemma

    pmi_df = pd.DataFrame(
        {
            "word_x": id_to_lemma[word_x_idx],
            "word_y": id_to_lemma[word_y_idx],
            "f_xy": f_xy.astype(np.int64),
            "f_x": f_x.astype(np.int64),
            "f_y": f_y.astype(np.int64),
            "pmi": pmi,
        }
    )
    pmi_df = pmi_df.sort_values(by="pmi", ascending=False, ignore_index=True)

    return pmi_df, dict(token_counts)


# --- Example Usage ------------------------------------------------------------
if __name__ == "__main__":
    from helper import load_sentences

    sentences_df = load_sentences(
        "eng-simple_wikipedia_2021_300K/eng-simple_wikipedia_2021_300K-sentences.txt"
    )

    pmi_results, word_frequencies = generate_lemmatized_pmi_fast(
        sentences_df, min_co_occurrence=5, skip_stopwords=False
    )

    # Caveat: with min_co_occurrence=0 the head of this ranking is dominated by
    # pairs that co-occur exactly once, which is PMI's known low-frequency bias.
    print("Top 10 word pairs with the highest PMI:")
    print(pmi_results.head(10))

    print("Bottom 10 word pairs with the lowest PMI:")
    print(pmi_results.tail(10))

    pmi_results.to_csv(
        "wordlists/lemmatized_pmi_results_300K_threshold_5_w_stopwords.csv",
        index=False,
        sep="\t",
        encoding="utf-8",
    )
    word_freq_df = pd.DataFrame(
        list(word_frequencies.items()), columns=["word", "frequency"]
    )
    word_freq_df.to_csv(
        "wordlists/lemmatized_word_frequencies_300K_threshold_5_w_stopwords.csv",
        index=False,
        sep="\t",
        encoding="utf-8",
    )